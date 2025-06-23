"""
Azure Deployment Templates
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class AzureTemplates:
    """Templates for Azure resource deployment"""

    @staticmethod
    def container_instance_template(
        container_name: str,
        image: str,
        location: str = "eastus",
        cpu: int = 1,
        memory: int = 1,
        port: int = 8000,
        environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Generate Container Instance ARM template"""

        template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "parameters": {
                "containerName": {
                    "type": "string",
                    "defaultValue": container_name,
                    "metadata": {"description": "Name for the container group"},
                },
                "image": {
                    "type": "string",
                    "defaultValue": image,
                    "metadata": {"description": "Container image"},
                },
                "location": {
                    "type": "string",
                    "defaultValue": location,
                    "metadata": {"description": "Location for all resources"},
                },
            },
            "resources": [
                {
                    "type": "Microsoft.ContainerInstance/containerGroups",
                    "apiVersion": "2021-03-01",
                    "name": "[parameters('containerName')]",
                    "location": "[parameters('location')]",
                    "properties": {
                        "containers": [
                            {
                                "name": container_name,
                                "properties": {
                                    "image": "[parameters('image')]",
                                    "resources": {
                                        "requests": {"cpu": cpu, "memoryInGb": memory}
                                    },
                                    "ports": [{"port": port, "protocol": "TCP"}],
                                    "environmentVariables": [
                                        {"name": k, "value": v}
                                        for k, v in (environment or {}).items()
                                    ],
                                },
                            }
                        ],
                        "osType": "Linux",
                        "restartPolicy": "Always",
                        "ipAddress": {
                            "type": "Public",
                            "ports": [{"port": port, "protocol": "TCP"}],
                        },
                    },
                }
            ],
            "outputs": {
                "containerIPv4Address": {
                    "type": "string",
                    "value": "[reference(resourceId('Microsoft.ContainerInstance/containerGroups', parameters('containerName'))).ipAddress.ip]",
                }
            },
        }

        return template

    @staticmethod
    def aks_cluster_template(
        cluster_name: str,
        location: str = "eastus",
        node_count: int = 3,
        vm_size: str = "Standard_DS2_v2",
        kubernetes_version: str = "1.24.0",
    ) -> Dict[str, Any]:
        """Generate AKS cluster ARM template"""

        template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "parameters": {
                "clusterName": {
                    "type": "string",
                    "defaultValue": cluster_name,
                    "metadata": {"description": "The name of the AKS cluster"},
                },
                "location": {
                    "type": "string",
                    "defaultValue": location,
                    "metadata": {"description": "The location of the AKS cluster"},
                },
                "nodeCount": {
                    "type": "int",
                    "defaultValue": node_count,
                    "minValue": 1,
                    "maxValue": 50,
                    "metadata": {"description": "The number of nodes for the cluster"},
                },
                "vmSize": {
                    "type": "string",
                    "defaultValue": vm_size,
                    "metadata": {"description": "The size of the Virtual Machine"},
                },
            },
            "resources": [
                {
                    "type": "Microsoft.ContainerService/managedClusters",
                    "apiVersion": "2021-05-01",
                    "name": "[parameters('clusterName')]",
                    "location": "[parameters('location')]",
                    "identity": {"type": "SystemAssigned"},
                    "properties": {
                        "dnsPrefix": "[concat(parameters('clusterName'), '-dns')]",
                        "kubernetesVersion": kubernetes_version,
                        "agentPoolProfiles": [
                            {
                                "name": "agentpool",
                                "osDiskSizeGB": 0,
                                "count": "[parameters('nodeCount')]",
                                "vmSize": "[parameters('vmSize')]",
                                "osType": "Linux",
                                "mode": "System",
                                "enableAutoScaling": True,
                                "minCount": 1,
                                "maxCount": 10,
                            }
                        ],
                        "networkProfile": {
                            "networkPlugin": "kubenet",
                            "loadBalancerSku": "standard",
                        },
                        "addonProfiles": {
                            "omsagent": {
                                "enabled": True,
                                "config": {
                                    "logAnalyticsWorkspaceResourceID": "[resourceId('Microsoft.OperationalInsights/workspaces', concat(parameters('clusterName'), '-workspace'))]"
                                },
                            }
                        },
                    },
                }
            ],
            "outputs": {
                "controlPlaneFQDN": {
                    "type": "string",
                    "value": "[reference(resourceId('Microsoft.ContainerService/managedClusters', parameters('clusterName'))).fqdn]",
                }
            },
        }

        return template

    @staticmethod
    def kubernetes_deployment_template(
        app_name: str,
        image: str,
        replicas: int = 3,
        cpu_request: str = "100m",
        memory_request: str = "128Mi",
        cpu_limit: str = "1000m",
        memory_limit: str = "512Mi",
    ) -> Dict[str, Any]:
        """Generate Kubernetes deployment template for AKS"""

        template = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": f"{app_name}-deployment", "labels": {"app": app_name}},
            "spec": {
                "replicas": replicas,
                "selector": {"matchLabels": {"app": app_name}},
                "template": {
                    "metadata": {"labels": {"app": app_name}},
                    "spec": {
                        "containers": [
                            {
                                "name": app_name,
                                "image": image,
                                "ports": [{"containerPort": 8000}],
                                "resources": {
                                    "requests": {
                                        "cpu": cpu_request,
                                        "memory": memory_request,
                                    },
                                    "limits": {
                                        "cpu": cpu_limit,
                                        "memory": memory_limit,
                                    },
                                },
                                "livenessProbe": {
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                },
                                "readinessProbe": {
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5,
                                },
                            }
                        ]
                    },
                },
            },
        }

        return template

    @staticmethod
    def kubernetes_service_template(app_name: str) -> Dict[str, Any]:
        """Generate Kubernetes service template"""

        template = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": f"{app_name}-service", "labels": {"app": app_name}},
            "spec": {
                "type": "LoadBalancer",
                "ports": [{"port": 80, "targetPort": 8000, "protocol": "TCP"}],
                "selector": {"app": app_name},
            },
        }

        return template

    @staticmethod
    def function_app_template(
        function_name: str,
        storage_account_name: str,
        location: str = "eastus",
        runtime: str = "python",
        runtime_version: str = "3.9",
    ) -> Dict[str, Any]:
        """Generate Azure Function App ARM template"""

        template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "parameters": {
                "functionAppName": {
                    "type": "string",
                    "defaultValue": function_name,
                    "metadata": {"description": "Name of the Function App"},
                },
                "storageAccountName": {
                    "type": "string",
                    "defaultValue": storage_account_name,
                    "metadata": {"description": "Name of the storage account"},
                },
                "location": {
                    "type": "string",
                    "defaultValue": location,
                    "metadata": {"description": "Location for all resources"},
                },
            },
            "variables": {
                "hostingPlanName": "[concat(parameters('functionAppName'), '-plan')]",
                "applicationInsightsName": "[concat(parameters('functionAppName'), '-insights')]",
            },
            "resources": [
                {
                    "type": "Microsoft.Storage/storageAccounts",
                    "apiVersion": "2021-04-01",
                    "name": "[parameters('storageAccountName')]",
                    "location": "[parameters('location')]",
                    "sku": {"name": "Standard_LRS"},
                    "kind": "StorageV2",
                },
                {
                    "type": "Microsoft.Web/serverfarms",
                    "apiVersion": "2021-02-01",
                    "name": "[variables('hostingPlanName')]",
                    "location": "[parameters('location')]",
                    "sku": {"name": "Y1", "tier": "Dynamic"},
                    "properties": {
                        "name": "[variables('hostingPlanName')]",
                        "computeMode": "Dynamic",
                    },
                },
                {
                    "type": "Microsoft.Insights/components",
                    "apiVersion": "2020-02-02",
                    "name": "[variables('applicationInsightsName')]",
                    "location": "[parameters('location')]",
                    "kind": "web",
                    "properties": {"Application_Type": "web", "Request_Source": "rest"},
                },
                {
                    "type": "Microsoft.Web/sites",
                    "apiVersion": "2021-02-01",
                    "name": "[parameters('functionAppName')]",
                    "location": "[parameters('location')]",
                    "kind": "functionapp",
                    "dependsOn": [
                        "[resourceId('Microsoft.Web/serverfarms', variables('hostingPlanName'))]",
                        "[resourceId('Microsoft.Storage/storageAccounts', parameters('storageAccountName'))]",
                        "[resourceId('Microsoft.Insights/components', variables('applicationInsightsName'))]",
                    ],
                    "properties": {
                        "serverFarmId": "[resourceId('Microsoft.Web/serverfarms', variables('hostingPlanName'))]",
                        "siteConfig": {
                            "appSettings": [
                                {
                                    "name": "AzureWebJobsStorage",
                                    "value": "[concat('DefaultEndpointsProtocol=https;AccountName=', parameters('storageAccountName'), ';EndpointSuffix=', environment().suffixes.storage, ';AccountKey=',listKeys(resourceId('Microsoft.Storage/storageAccounts', parameters('storageAccountName')), '2021-04-01').keys[0].value)]",
                                },
                                {
                                    "name": "WEBSITE_CONTENTAZUREFILECONNECTIONSTRING",
                                    "value": "[concat('DefaultEndpointsProtocol=https;AccountName=', parameters('storageAccountName'), ';EndpointSuffix=', environment().suffixes.storage, ';AccountKey=',listKeys(resourceId('Microsoft.Storage/storageAccounts', parameters('storageAccountName')), '2021-04-01').keys[0].value)]",
                                },
                                {
                                    "name": "WEBSITE_CONTENTSHARE",
                                    "value": "[toLower(parameters('functionAppName'))]",
                                },
                                {"name": "FUNCTIONS_EXTENSION_VERSION", "value": "~4"},
                                {"name": "FUNCTIONS_WORKER_RUNTIME", "value": runtime},
                                {
                                    "name": "APPINSIGHTS_INSTRUMENTATIONKEY",
                                    "value": "[reference(resourceId('Microsoft.Insights/components', variables('applicationInsightsName'))).InstrumentationKey]",
                                },
                            ],
                            "pythonVersion": runtime_version
                            if runtime == "python"
                            else None,
                        },
                    },
                },
            ],
            "outputs": {
                "functionAppHostName": {
                    "type": "string",
                    "value": "[reference(resourceId('Microsoft.Web/sites', parameters('functionAppName'))).defaultHostName]",
                }
            },
        }

        return template

    @staticmethod
    def storage_account_template(
        storage_account_name: str,
        location: str = "eastus",
        sku: str = "Standard_LRS",
        tier: str = "Hot",
    ) -> Dict[str, Any]:
        """Generate Storage Account ARM template"""

        template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "parameters": {
                "storageAccountName": {
                    "type": "string",
                    "defaultValue": storage_account_name,
                    "metadata": {"description": "Name of the storage account"},
                },
                "location": {
                    "type": "string",
                    "defaultValue": location,
                    "metadata": {"description": "Location for the storage account"},
                },
            },
            "resources": [
                {
                    "type": "Microsoft.Storage/storageAccounts",
                    "apiVersion": "2021-04-01",
                    "name": "[parameters('storageAccountName')]",
                    "location": "[parameters('location')]",
                    "sku": {"name": sku},
                    "kind": "StorageV2",
                    "properties": {
                        "accessTier": tier,
                        "supportsHttpsTrafficOnly": True,
                        "minimumTlsVersion": "TLS1_2",
                    },
                }
            ],
            "outputs": {
                "storageAccountId": {
                    "type": "string",
                    "value": "[resourceId('Microsoft.Storage/storageAccounts', parameters('storageAccountName'))]",
                }
            },
        }

        return template

    @staticmethod
    def virtual_network_template(
        vnet_name: str,
        location: str = "eastus",
        address_space: str = "10.0.0.0/16",
        subnets: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Generate Virtual Network ARM template"""

        if subnets is None:
            subnets = [{"name": "default", "addressPrefix": "10.0.0.0/24"}]

        template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "parameters": {
                "vnetName": {
                    "type": "string",
                    "defaultValue": vnet_name,
                    "metadata": {"description": "Name of the Virtual Network"},
                },
                "location": {
                    "type": "string",
                    "defaultValue": location,
                    "metadata": {"description": "Location for the Virtual Network"},
                },
            },
            "resources": [
                {
                    "type": "Microsoft.Network/virtualNetworks",
                    "apiVersion": "2021-02-01",
                    "name": "[parameters('vnetName')]",
                    "location": "[parameters('location')]",
                    "properties": {
                        "addressSpace": {"addressPrefixes": [address_space]},
                        "subnets": [
                            {
                                "name": subnet["name"],
                                "properties": {
                                    "addressPrefix": subnet["addressPrefix"]
                                },
                            }
                            for subnet in subnets
                        ],
                    },
                }
            ],
            "outputs": {
                "vnetId": {
                    "type": "string",
                    "value": "[resourceId('Microsoft.Network/virtualNetworks', parameters('vnetName'))]",
                }
            },
        }

        return template

    @staticmethod
    def azure_sql_template(
        server_name: str,
        database_name: str,
        location: str = "eastus",
        admin_user: str = "sqladmin",
        tier: str = "Basic",
    ) -> Dict[str, Any]:
        """Generate Azure SQL Database ARM template"""

        template = {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "parameters": {
                "serverName": {
                    "type": "string",
                    "defaultValue": server_name,
                    "metadata": {"description": "Name of the SQL Server"},
                },
                "databaseName": {
                    "type": "string",
                    "defaultValue": database_name,
                    "metadata": {"description": "Name of the SQL Database"},
                },
                "location": {
                    "type": "string",
                    "defaultValue": location,
                    "metadata": {"description": "Location for all resources"},
                },
                "administratorLogin": {
                    "type": "string",
                    "defaultValue": admin_user,
                    "metadata": {
                        "description": "Administrator username for the SQL Server"
                    },
                },
                "administratorLoginPassword": {
                    "type": "securestring",
                    "metadata": {
                        "description": "Administrator password for the SQL Server"
                    },
                },
            },
            "resources": [
                {
                    "type": "Microsoft.Sql/servers",
                    "apiVersion": "2021-02-01-preview",
                    "name": "[parameters('serverName')]",
                    "location": "[parameters('location')]",
                    "properties": {
                        "administratorLogin": "[parameters('administratorLogin')]",
                        "administratorLoginPassword": "[parameters('administratorLoginPassword')]",
                        "version": "12.0",
                    },
                },
                {
                    "type": "Microsoft.Sql/servers/databases",
                    "apiVersion": "2021-02-01-preview",
                    "name": "[concat(parameters('serverName'), '/', parameters('databaseName'))]",
                    "location": "[parameters('location')]",
                    "dependsOn": [
                        "[resourceId('Microsoft.Sql/servers', parameters('serverName'))]"
                    ],
                    "sku": {"name": tier, "tier": tier},
                    "properties": {"collation": "SQL_Latin1_General_CP1_CI_AS"},
                },
                {
                    "type": "Microsoft.Sql/servers/firewallRules",
                    "apiVersion": "2021-02-01-preview",
                    "name": "[concat(parameters('serverName'), '/AllowAllWindowsAzureIps')]",
                    "dependsOn": [
                        "[resourceId('Microsoft.Sql/servers', parameters('serverName'))]"
                    ],
                    "properties": {
                        "startIpAddress": "0.0.0.0",
                        "endIpAddress": "0.0.0.0",
                    },
                },
            ],
            "outputs": {
                "sqlServerFqdn": {
                    "type": "string",
                    "value": "[reference(resourceId('Microsoft.Sql/servers', parameters('serverName'))).fullyQualifiedDomainName]",
                }
            },
        }

        return template

    @staticmethod
    def bicep_main_template() -> str:
        """Generate main Bicep template"""

        template = """@description('Subscription ID')
param subscriptionId string = subscription().subscriptionId

@description('Resource Group Name')
param resourceGroupName string

@description('Location for all resources')
param location string = resourceGroup().location

@description('Environment name (dev, staging, prod)')
param environment string = 'dev'

@description('Application name prefix')
param appName string = 'inferloop-synthetic'

// Variables
var storageAccountName = '${appName}${environment}storage'
var containerInstanceName = '${appName}-${environment}-api'
var aksClusterName = '${appName}-${environment}-aks'
var sqlServerName = '${appName}-${environment}-sql'
var keyVaultName = '${appName}-${environment}-kv'

// Storage Account
resource storageAccount 'Microsoft.Storage/storageAccounts@2021-04-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
  }
}

// Container Instance
resource containerInstance 'Microsoft.ContainerInstance/containerGroups@2021-03-01' = {
  name: containerInstanceName
  location: location
  properties: {
    containers: [
      {
        name: containerInstanceName
        properties: {
          image: 'mcr.microsoft.com/azuredocs/aci-helloworld'
          resources: {
            requests: {
              cpu: 1
              memoryInGb: 1
            }
          }
          ports: [
            {
              port: 8000
              protocol: 'TCP'
            }
          ]
        }
      }
    ]
    osType: 'Linux'
    restartPolicy: 'Always'
    ipAddress: {
      type: 'Public'
      ports: [
        {
          port: 8000
          protocol: 'TCP'
        }
      ]
    }
  }
}

// AKS Cluster
resource aksCluster 'Microsoft.ContainerService/managedClusters@2021-05-01' = {
  name: aksClusterName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    dnsPrefix: '${aksClusterName}-dns'
    agentPoolProfiles: [
      {
        name: 'agentpool'
        count: 3
        vmSize: 'Standard_DS2_v2'
        osType: 'Linux'
        mode: 'System'
        enableAutoScaling: true
        minCount: 1
        maxCount: 10
      }
    ]
    networkProfile: {
      networkPlugin: 'kubenet'
      loadBalancerSku: 'standard'
    }
  }
}

// SQL Server
resource sqlServer 'Microsoft.Sql/servers@2021-02-01-preview' = {
  name: sqlServerName
  location: location
  properties: {
    administratorLogin: 'sqladmin'
    administratorLoginPassword: 'ComplexP@ssw0rd!'
    version: '12.0'
  }
}

// SQL Database
resource sqlDatabase 'Microsoft.Sql/servers/databases@2021-02-01-preview' = {
  parent: sqlServer
  name: 'synthetic_data'
  location: location
  sku: {
    name: 'Basic'
    tier: 'Basic'
  }
  properties: {
    collation: 'SQL_Latin1_General_CP1_CI_AS'
  }
}

// Key Vault
resource keyVault 'Microsoft.KeyVault/vaults@2021-04-01-preview' = {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    accessPolicies: []
  }
}

// Outputs
output storageAccountName string = storageAccount.name
output containerInstanceIp string = containerInstance.properties.ipAddress.ip
output aksClusterName string = aksCluster.name
output sqlServerFqdn string = sqlServer.properties.fullyQualifiedDomainName
output keyVaultUri string = keyVault.properties.vaultUri
"""

        return template

    @staticmethod
    def deployment_script_template() -> str:
        """Generate Azure deployment script"""

        script = """#!/bin/bash
# Azure Deployment Script for Inferloop Synthetic Data

set -e

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

# Configuration
SUBSCRIPTION_ID="${SUBSCRIPTION_ID}"
RESOURCE_GROUP="${RESOURCE_GROUP:-inferloop-rg}"
LOCATION="${LOCATION:-eastus}"
ENVIRONMENT="${ENVIRONMENT:-dev}"

echo -e "${GREEN}Starting Azure deployment for Inferloop Synthetic Data${NC}"

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        echo -e "${RED}Azure CLI not found. Please install it first.${NC}"
        exit 1
    fi
    
    # Check Bicep (optional)
    if command -v az bicep version &> /dev/null; then
        echo -e "${GREEN}Bicep found${NC}"
        USE_BICEP=true
    else
        echo -e "${YELLOW}Bicep not found. Using ARM templates.${NC}"
        USE_BICEP=false
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${YELLOW}kubectl not found. AKS deployment will be limited.${NC}"
    fi
}

# Login and set subscription
azure_login() {
    echo -e "${YELLOW}Setting up Azure authentication...${NC}"
    
    # Check if already logged in
    if ! az account show &> /dev/null; then
        az login
    fi
    
    # Set subscription
    az account set --subscription "${SUBSCRIPTION_ID}"
    
    # Create resource group
    az group create --name "${RESOURCE_GROUP}" --location "${LOCATION}"
}

# Deploy with Bicep
deploy_with_bicep() {
    echo -e "${YELLOW}Deploying with Bicep...${NC}"
    
    az deployment group create \\
        --resource-group "${RESOURCE_GROUP}" \\
        --template-file bicep/main.bicep \\
        --parameters \\
            subscriptionId="${SUBSCRIPTION_ID}" \\
            resourceGroupName="${RESOURCE_GROUP}" \\
            location="${LOCATION}" \\
            environment="${ENVIRONMENT}"
}

# Deploy with ARM templates
deploy_with_arm() {
    echo -e "${YELLOW}Deploying with ARM templates...${NC}"
    
    # Deploy storage account
    az deployment group create \\
        --resource-group "${RESOURCE_GROUP}" \\
        --template-file arm/storage.json \\
        --parameters \\
            storageAccountName="inferloop${ENVIRONMENT}storage" \\
            location="${LOCATION}"
    
    # Deploy container instance
    az deployment group create \\
        --resource-group "${RESOURCE_GROUP}" \\
        --template-file arm/container_instance.json \\
        --parameters \\
            containerName="inferloop-${ENVIRONMENT}-api" \\
            location="${LOCATION}"
    
    # Deploy AKS cluster
    read -p "Deploy AKS cluster? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        az deployment group create \\
            --resource-group "${RESOURCE_GROUP}" \\
            --template-file arm/aks.json \\
            --parameters \\
                clusterName="inferloop-${ENVIRONMENT}-aks" \\
                location="${LOCATION}"
    fi
    
    # Deploy SQL Database
    read -p "Deploy SQL Database? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -s -p "Enter SQL admin password: " SQL_PASSWORD
        echo
        
        az deployment group create \\
            --resource-group "${RESOURCE_GROUP}" \\
            --template-file arm/sql.json \\
            --parameters \\
                serverName="inferloop-${ENVIRONMENT}-sql" \\
                databaseName="synthetic_data" \\
                location="${LOCATION}" \\
                administratorLoginPassword="${SQL_PASSWORD}"
    fi
}

# Configure AKS
configure_aks() {
    echo -e "${YELLOW}Configuring AKS cluster...${NC}"
    
    CLUSTER_NAME="inferloop-${ENVIRONMENT}-aks"
    
    # Get AKS credentials
    az aks get-credentials \\
        --resource-group "${RESOURCE_GROUP}" \\
        --name "${CLUSTER_NAME}"
    
    # Deploy to AKS
    if command -v kubectl &> /dev/null; then
        kubectl apply -f kubernetes/
    fi
}

# Build and push container image
build_and_push_image() {
    echo -e "${YELLOW}Building and pushing container image...${NC}"
    
    REGISTRY_NAME="inferloop${ENVIRONMENT}registry"
    IMAGE_NAME="inferloop-synthetic"
    
    # Create container registry
    az acr create \\
        --resource-group "${RESOURCE_GROUP}" \\
        --name "${REGISTRY_NAME}" \\
        --sku Basic \\
        --location "${LOCATION}"
    
    # Build and push image
    az acr build \\
        --registry "${REGISTRY_NAME}" \\
        --image "${IMAGE_NAME}:latest" \\
        .
}

# Main deployment
main() {
    check_prerequisites
    azure_login
    
    # Optional: Build and push image
    read -p "Build and push Docker image? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        build_and_push_image
    fi
    
    if [ "${USE_BICEP}" = true ]; then
        deploy_with_bicep
    else
        deploy_with_arm
    fi
    
    # Optional: Configure AKS
    read -p "Configure AKS cluster? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        configure_aks
    fi
    
    echo -e "${GREEN}Deployment completed successfully!${NC}"
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Update container images with your application"
    echo "2. Configure environment variables and secrets"
    echo "3. Set up monitoring and alerts"
    echo "4. Configure custom domains and SSL certificates"
}

# Run main function
main
"""

        return script

    @staticmethod
    def dockerfile_template() -> str:
        """Generate Dockerfile for Azure deployment"""

        dockerfile = """# Multi-stage build for Azure Container Instances
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml setup.py ./
COPY inferloop_synthetic ./inferloop_synthetic

# Install dependencies
RUN pip install --no-cache-dir -e ".[all]"

# Production stage
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app /app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Environment variables for Azure
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV AZURE_CLIENT_ID=""
ENV AZURE_CLIENT_SECRET=""
ENV AZURE_TENANT_ID=""

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "inferloop_synthetic.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

        return dockerfile

    @staticmethod
    def save_templates(output_dir: Path):
        """Save all Azure templates to files"""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_dir / "arm").mkdir(exist_ok=True)
        (output_dir / "bicep").mkdir(exist_ok=True)
        (output_dir / "kubernetes").mkdir(exist_ok=True)
        (output_dir / "scripts").mkdir(exist_ok=True)

        # Save Bicep template
        with open(output_dir / "bicep" / "main.bicep", "w") as f:
            f.write(AzureTemplates.bicep_main_template())

        # Save ARM templates
        container_template = AzureTemplates.container_instance_template(
            "inferloop-synthetic", "mcr.microsoft.com/azuredocs/aci-helloworld"
        )
        with open(output_dir / "arm" / "container_instance.json", "w") as f:
            json.dump(container_template, f, indent=2)

        aks_template = AzureTemplates.aks_cluster_template("inferloop-aks")
        with open(output_dir / "arm" / "aks.json", "w") as f:
            json.dump(aks_template, f, indent=2)

        storage_template = AzureTemplates.storage_account_template(
            "inferloopdatastorage"
        )
        with open(output_dir / "arm" / "storage.json", "w") as f:
            json.dump(storage_template, f, indent=2)

        sql_template = AzureTemplates.azure_sql_template(
            "inferloop-sql", "synthetic_data"
        )
        with open(output_dir / "arm" / "sql.json", "w") as f:
            json.dump(sql_template, f, indent=2)

        function_template = AzureTemplates.function_app_template(
            "inferloop-functions", "inferloopdatastorage"
        )
        with open(output_dir / "arm" / "function_app.json", "w") as f:
            json.dump(function_template, f, indent=2)

        vnet_template = AzureTemplates.virtual_network_template("inferloop-vnet")
        with open(output_dir / "arm" / "vnet.json", "w") as f:
            json.dump(vnet_template, f, indent=2)

        # Save Kubernetes templates
        k8s_deployment = AzureTemplates.kubernetes_deployment_template(
            "inferloop-synthetic",
            "inferloopregistry.azurecr.io/inferloop-synthetic:latest",
        )
        with open(output_dir / "kubernetes" / "deployment.yaml", "w") as f:
            yaml.dump(k8s_deployment, f, default_flow_style=False)

        k8s_service = AzureTemplates.kubernetes_service_template("inferloop-synthetic")
        with open(output_dir / "kubernetes" / "service.yaml", "w") as f:
            yaml.dump(k8s_service, f, default_flow_style=False)

        # Save deployment script
        with open(output_dir / "scripts" / "deploy.sh", "w") as f:
            f.write(AzureTemplates.deployment_script_template())

        # Make script executable
        (output_dir / "scripts" / "deploy.sh").chmod(0o755)

        # Save Dockerfile
        with open(output_dir / "Dockerfile", "w") as f:
            f.write(AzureTemplates.dockerfile_template())

        # Save example parameters files
        examples_dir = output_dir / "examples"
        examples_dir.mkdir(exist_ok=True)

        # Parameters example
        params_example = {
            "parameters": {
                "subscriptionId": {"value": "your-subscription-id"},
                "resourceGroupName": {"value": "inferloop-rg"},
                "location": {"value": "eastus"},
                "environment": {"value": "dev"},
            }
        }
        with open(examples_dir / "parameters.json", "w") as f:
            json.dump(params_example, f, indent=2)
