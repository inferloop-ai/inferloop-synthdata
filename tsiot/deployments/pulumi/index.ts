import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";
import * as awsx from "@pulumi/awsx";
import * as eks from "@pulumi/eks";
import * as k8s from "@pulumi/kubernetes";

// Get configuration
const config = new pulumi.Config();
const awsConfig = new pulumi.Config("aws");

// Project configuration
const projectName = config.require("projectName");
const environment = config.require("environment");
const owner = config.get("owner") || "tsiot-team";
const costCenter = config.get("costCenter") || "engineering";

// AWS configuration
const region = awsConfig.require("region");

// Tags
const commonTags = {
    Project: projectName,
    Environment: environment,
    Owner: owner,
    CostCenter: costCenter,
    ManagedBy: "pulumi",
};

// Networking configuration
const vpcCidr = config.get("vpcCidr") || "10.0.0.0/16";
const availabilityZones = config.getObject<string[]>("availabilityZones") || ["us-west-2a", "us-west-2b"];
const privateSubnetCidrs = config.getObject<string[]>("privateSubnetCidrs") || ["10.0.1.0/24", "10.0.2.0/24"];
const publicSubnetCidrs = config.getObject<string[]>("publicSubnetCidrs") || ["10.0.101.0/24", "10.0.102.0/24"];
const databaseSubnetCidrs = config.getObject<string[]>("databaseSubnetCidrs") || ["10.0.201.0/24", "10.0.202.0/24"];

// Feature flags
const enableEncryption = config.getBoolean("enableEncryption") || false;
const enableMonitoring = config.getBoolean("enableMonitoring") || true;
const enableGpuNodes = config.getBoolean("enableGpuNodes") || false;

// Create KMS key for encryption
const kmsKey = new aws.kms.Key(`${projectName}-${environment}-key`, {
    description: `KMS key for ${projectName} ${environment} encryption`,
    enableKeyRotation: true,
    tags: commonTags,
});

const kmsAlias = new aws.kms.Alias(`${projectName}-${environment}-alias`, {
    name: `alias/${projectName}-${environment}`,
    targetKeyId: kmsKey.keyId,
});

// Create VPC
const vpc = new awsx.ec2.Vpc(`${projectName}-${environment}-vpc`, {
    cidrBlock: vpcCidr,
    numberOfAvailabilityZones: availabilityZones.length,
    numberOfNatGateways: environment === "prod" ? availabilityZones.length : 1,
    enableDnsHostnames: true,
    enableDnsSupport: true,
    tags: {
        ...commonTags,
        Name: `${projectName}-${environment}-vpc`,
    },
});

// Create EKS cluster
const clusterName = config.get("clusterName") || `${projectName}-${environment}-cluster`;
const clusterVersion = config.get("clusterVersion") || "1.28";

const cluster = new eks.Cluster(`${projectName}-${environment}-cluster`, {
    name: clusterName,
    version: clusterVersion,
    vpcId: vpc.vpcId,
    subnetIds: vpc.privateSubnetIds,
    instanceType: "m5.large",
    desiredCapacity: 2,
    minSize: 1,
    maxSize: 10,
    nodeAssociatePublicIpAddress: false,
    endpointPrivateAccess: true,
    endpointPublicAccess: true,
    publicAccessCidrs: config.getObject<string[]>("allowedCidrBlocks") || ["0.0.0.0/0"],
    
    // Enable encryption if configured
    encryptionConfigKeyArn: enableEncryption ? kmsKey.arn : undefined,
    
    // Enable cluster logging
    enabledClusterLogTypes: ["api", "audit", "authenticator", "controllerManager", "scheduler"],
    
    tags: commonTags,
});

// Create additional node groups
const nodeGroups = config.getObject<any>("nodeGroups") || {};

const managedNodeGroups: { [key: string]: eks.ManagedNodeGroup } = {};

Object.entries(nodeGroups).forEach(([name, nodeGroupConfig]: [string, any]) => {
    managedNodeGroups[name] = new eks.ManagedNodeGroup(`${projectName}-${environment}-${name}`, {
        cluster: cluster.core,
        nodeGroupName: `${projectName}-${environment}-${name}`,
        instanceTypes: nodeGroupConfig.instanceTypes,
        scalingConfig: nodeGroupConfig.scalingConfig,
        diskSize: nodeGroupConfig.diskSize,
        capacityType: nodeGroupConfig.capacityType,
        labels: nodeGroupConfig.labels || {},
        taints: nodeGroupConfig.taints || [],
        tags: commonTags,
    });
});

// Create GPU node group if enabled
let gpuNodeGroup: eks.ManagedNodeGroup | undefined;
if (enableGpuNodes) {
    const gpuConfig = config.getObject<any>("gpuNodeConfig") || {
        instanceTypes: ["g4dn.xlarge"],
        scalingConfig: { desiredSize: 0, maxSize: 2, minSize: 0 }
    };
    
    gpuNodeGroup = new eks.ManagedNodeGroup(`${projectName}-${environment}-gpu`, {
        cluster: cluster.core,
        nodeGroupName: `${projectName}-${environment}-gpu`,
        instanceTypes: gpuConfig.instanceTypes,
        scalingConfig: gpuConfig.scalingConfig,
        diskSize: 100,
        capacityType: "ON_DEMAND",
        labels: {
            "node-type": "gpu",
            "nvidia.com/gpu": "true",
        },
        taints: [{
            key: "nvidia.com/gpu",
            value: "true",
            effect: "NO_SCHEDULE",
        }],
        tags: {
            ...commonTags,
            NodeType: "gpu",
        },
    });
}

// RDS Database
const rdsSubnetGroup = new aws.rds.SubnetGroup(`${projectName}-${environment}-rds`, {
    name: `${projectName}-${environment}-rds`,
    subnetIds: vpc.privateSubnetIds.apply(ids => ids.slice(0, Math.min(ids.length, databaseSubnetCidrs.length))),
    tags: commonTags,
});

// RDS Security Group
const rdsSecurityGroup = new aws.ec2.SecurityGroup(`${projectName}-${environment}-rds`, {
    name: `${projectName}-${environment}-rds`,
    vpcId: vpc.vpcId,
    ingress: [{
        fromPort: 5432,
        toPort: 5432,
        protocol: "tcp",
        securityGroups: [cluster.core.cluster.vpcConfig.clusterSecurityGroupId],
    }],
    egress: [{
        fromPort: 0,
        toPort: 0,
        protocol: "-1",
        cidrBlocks: ["0.0.0.0/0"],
    }],
    tags: commonTags,
});

// Random password for RDS
const rdsPassword = new aws.rds.SubnetGroup(`${projectName}-${environment}-rds-password`, {
    name: `${projectName}-${environment}-rds-password`,
    subnetIds: [vpc.privateSubnetIds.apply(ids => ids[0])], // Dummy, will be replaced
});

const rdsInstance = new aws.rds.Instance(`${projectName}-${environment}-rds`, {
    identifier: `${projectName}-${environment}-rds`,
    engine: "postgres",
    engineVersion: config.get("rdsEngineVersion") || "15.4",
    instanceClass: config.get("rdsInstanceClass") || "db.t3.medium",
    allocatedStorage: config.getNumber("rdsAllocatedStorage") || 100,
    maxAllocatedStorage: config.getNumber("rdsMaxAllocatedStorage") || 1000,
    
    dbName: "tsiot",
    username: "tsiot_admin",
    password: "temp_password_will_be_rotated", // This should be rotated after creation
    
    dbSubnetGroupName: rdsSubnetGroup.name,
    vpcSecurityGroupIds: [rdsSecurityGroup.id],
    
    backupRetentionPeriod: config.getNumber("rdsBackupRetentionPeriod") || 7,
    backupWindow: "03:00-04:00",
    maintenanceWindow: "sun:04:00-sun:05:00",
    multiAz: config.getBoolean("rdsMultiAz") || false,
    
    storageEncrypted: enableEncryption,
    kmsKeyId: enableEncryption ? kmsKey.arn : undefined,
    
    performanceInsightsEnabled: environment === "prod",
    performanceInsightsRetentionPeriod: environment === "prod" ? 31 : 7,
    
    skipFinalSnapshot: environment !== "prod",
    deletionProtection: environment === "prod",
    
    tags: commonTags,
});

// ElastiCache Redis
const redisSubnetGroup = new aws.elasticache.SubnetGroup(`${projectName}-${environment}-redis`, {
    name: `${projectName}-${environment}-redis`,
    subnetIds: vpc.privateSubnetIds,
    tags: commonTags,
});

const redisSecurityGroup = new aws.ec2.SecurityGroup(`${projectName}-${environment}-redis`, {
    name: `${projectName}-${environment}-redis`,
    vpcId: vpc.vpcId,
    ingress: [{
        fromPort: 6379,
        toPort: 6379,
        protocol: "tcp",
        securityGroups: [cluster.core.cluster.vpcConfig.clusterSecurityGroupId],
    }],
    tags: commonTags,
});

const redisCluster = new aws.elasticache.ReplicationGroup(`${projectName}-${environment}-redis`, {
    replicationGroupId: `${projectName}-${environment}-redis`,
    description: `Redis cluster for ${projectName} ${environment}`,
    
    nodeType: config.get("redisNodeType") || "cache.t3.micro",
    numCacheCluster: config.getNumber("redisNumCacheNodes") || 1,
    port: 6379,
    parameterGroupName: "default.redis7",
    engineVersion: config.get("redisEngineVersion") || "7.0",
    
    subnetGroupName: redisSubnetGroup.name,
    securityGroupIds: [redisSecurityGroup.id],
    
    atRestEncryptionEnabled: enableEncryption,
    transitEncryptionEnabled: enableEncryption,
    authTokenEnabled: enableEncryption,
    
    automaticFailoverEnabled: true,
    multiAzEnabled: environment === "prod",
    
    snapshotRetentionLimit: 5,
    snapshotWindow: "03:00-05:00",
    
    tags: commonTags,
});

// MSK (Kafka)
const mskSecurityGroup = new aws.ec2.SecurityGroup(`${projectName}-${environment}-msk`, {
    name: `${projectName}-${environment}-msk`,
    vpcId: vpc.vpcId,
    ingress: [
        {
            fromPort: 9092,
            toPort: 9092,
            protocol: "tcp",
            securityGroups: [cluster.core.cluster.vpcConfig.clusterSecurityGroupId],
        },
        {
            fromPort: 9094,
            toPort: 9094,
            protocol: "tcp",
            securityGroups: [cluster.core.cluster.vpcConfig.clusterSecurityGroupId],
        },
        {
            fromPort: 2181,
            toPort: 2181,
            protocol: "tcp",
            securityGroups: [cluster.core.cluster.vpcConfig.clusterSecurityGroupId],
        },
    ],
    tags: commonTags,
});

const mskCluster = new aws.msk.Cluster(`${projectName}-${environment}-kafka`, {
    clusterName: `${projectName}-${environment}-kafka`,
    kafkaVersion: config.get("kafkaVersion") || "3.5.1",
    numberOfBrokerNodes: availabilityZones.length,
    
    brokerNodeGroupInfo: {
        instanceType: config.get("kafkaInstanceType") || "kafka.t3.small",
        clientSubnets: vpc.privateSubnetIds,
        securityGroups: [mskSecurityGroup.id],
        storageInfo: {
            ebsStorageInfo: {
                volumeSize: config.getNumber("kafkaEbsVolumeSize") || 100,
            },
        },
    },
    
    clientAuthentication: {
        tls: {
            certificateAuthorityArns: [],
        },
        sasl: {
            scram: true,
        },
    },
    
    encryptionInfo: {
        encryptionAtRestKmsKeyArn: enableEncryption ? kmsKey.arn : undefined,
        encryptionInTransit: {
            clientBroker: "TLS",
            inCluster: true,
        },
    },
    
    loggingInfo: {
        brokerLogs: {
            cloudwatchLogs: {
                enabled: true,
                logGroup: `/aws/msk/${projectName}-${environment}`,
            },
        },
    },
    
    tags: commonTags,
});

// S3 Buckets
const dataBucket = new aws.s3.Bucket(`${projectName}-${environment}-data`, {
    bucket: `${projectName}-${environment}-data-${Date.now()}`,
    versioning: {
        enabled: true,
    },
    serverSideEncryptionConfiguration: enableEncryption ? {
        rule: {
            applyServerSideEncryptionByDefault: {
                sseAlgorithm: "aws:kms",
                kmsMasterKeyId: kmsKey.arn,
            },
        },
    } : undefined,
    lifecycleRules: environment === "prod" ? [{
        id: "transition_to_ia",
        enabled: true,
        transitions: [
            {
                days: 30,
                storageClass: "STANDARD_IA",
            },
            {
                days: 90,
                storageClass: "GLACIER",
            },
            {
                days: 365,
                storageClass: "DEEP_ARCHIVE",
            },
        ],
    }] : undefined,
    tags: commonTags,
});

const backupBucket = new aws.s3.Bucket(`${projectName}-${environment}-backup`, {
    bucket: `${projectName}-${environment}-backup-${Date.now()}`,
    versioning: {
        enabled: true,
    },
    serverSideEncryptionConfiguration: enableEncryption ? {
        rule: {
            applyServerSideEncryptionByDefault: {
                sseAlgorithm: "aws:kms",
                kmsMasterKeyId: kmsKey.arn,
            },
        },
    } : undefined,
    tags: commonTags,
});

// Kubernetes provider
const k8sProvider = new k8s.Provider(`${projectName}-${environment}-k8s`, {
    kubeconfig: cluster.kubeconfig,
});

// Install AWS Load Balancer Controller
const awsLoadBalancerController = new k8s.helm.v3.Chart(`${projectName}-${environment}-aws-load-balancer-controller`, {
    chart: "aws-load-balancer-controller",
    repository: "https://aws.github.io/eks-charts",
    namespace: "kube-system",
    values: {
        clusterName: cluster.core.cluster.name,
        serviceAccount: {
            create: true,
            name: "aws-load-balancer-controller",
            annotations: {
                "eks.amazonaws.com/role-arn": cluster.core.instanceRoles.apply(roles => roles[0].arn),
            },
        },
    },
}, { provider: k8sProvider });

// Install Cluster Autoscaler
const clusterAutoscaler = new k8s.helm.v3.Chart(`${projectName}-${environment}-cluster-autoscaler`, {
    chart: "cluster-autoscaler",
    repository: "https://kubernetes.github.io/autoscaler",
    namespace: "kube-system",
    values: {
        autoDiscovery: {
            clusterName: cluster.core.cluster.name,
        },
        awsRegion: region,
        serviceAccount: {
            annotations: {
                "eks.amazonaws.com/role-arn": cluster.core.instanceRoles.apply(roles => roles[0].arn),
            },
        },
    },
}, { provider: k8sProvider });

// Install monitoring stack if enabled
let monitoring: k8s.helm.v3.Chart | undefined;
if (enableMonitoring) {
    monitoring = new k8s.helm.v3.Chart(`${projectName}-${environment}-monitoring`, {
        chart: "kube-prometheus-stack",
        repository: "https://prometheus-community.github.io/helm-charts",
        namespace: "monitoring",
        values: {
            prometheus: {
                prometheusSpec: {
                    storageSpec: {
                        volumeClaimTemplate: {
                            spec: {
                                storageClassName: "gp3",
                                accessModes: ["ReadWriteOnce"],
                                resources: {
                                    requests: {
                                        storage: "50Gi",
                                    },
                                },
                            },
                        },
                    },
                },
            },
            grafana: {
                persistence: {
                    enabled: true,
                    storageClassName: "gp3",
                    size: "10Gi",
                },
                adminPassword: "admin", // Should be changed in production
            },
        },
    }, { provider: k8sProvider });
}

// Outputs
export const vpcId = vpc.vpcId;
export const privateSubnetIds = vpc.privateSubnetIds;
export const publicSubnetIds = vpc.publicSubnetIds;
export const clusterName_output = cluster.core.cluster.name;
export const clusterEndpoint = cluster.core.cluster.endpoint;
export const kubeconfig = cluster.kubeconfig;
export const rdsEndpoint = rdsInstance.endpoint;
export const redisEndpoint = redisCluster.primaryEndpointAddress;
export const kafkaBootstrapServers = mskCluster.bootstrapBrokersTls;
export const dataBucketName = dataBucket.bucket;
export const backupBucketName = backupBucket.bucket;
export const kmsKeyId = kmsKey.keyId;
export const kmsKeyArn = kmsKey.arn;