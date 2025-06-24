"""Helm chart generation and management for on-premises deployments."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import subprocess
import shutil

from ..base import ResourceConfig, DeploymentResult
from ..exceptions import DeploymentError


class HelmChartGenerator:
    """Generate and manage Helm charts for synthetic data deployments."""
    
    def __init__(self, chart_name: str = "synthdata", version: str = "0.1.0"):
        self.chart_name = chart_name
        self.version = version
        self.chart_dir = None
        
    def create_chart(self, config: ResourceConfig, output_dir: Optional[str] = None) -> str:
        """Create a Helm chart from resource configuration.
        
        Args:
            config: Resource configuration
            output_dir: Output directory for the chart
            
        Returns:
            Path to the created chart directory
        """
        if output_dir:
            self.chart_dir = Path(output_dir) / self.chart_name
        else:
            self.chart_dir = Path(tempfile.mkdtemp()) / self.chart_name
            
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        
        # Create chart structure
        self._create_chart_yaml()
        self._create_values_yaml(config)
        self._create_templates(config)
        self._create_helpers()
        
        return str(self.chart_dir)
    
    def _create_chart_yaml(self):
        """Create Chart.yaml file."""
        chart_metadata = {
            "apiVersion": "v2",
            "name": self.chart_name,
            "description": "Helm chart for Inferloop Synthetic Data Platform",
            "type": "application",
            "version": self.version,
            "appVersion": "1.0.0",
            "keywords": ["synthetic-data", "ml", "data-generation"],
            "home": "https://inferloop.com",
            "maintainers": [
                {
                    "name": "Inferloop Team",
                    "email": "support@inferloop.com"
                }
            ],
            "dependencies": []
        }
        
        with open(self.chart_dir / "Chart.yaml", "w") as f:
            yaml.dump(chart_metadata, f, default_flow_style=False)
    
    def _create_values_yaml(self, config: ResourceConfig):
        """Create values.yaml with default configuration."""
        values = {
            "replicaCount": config.compute.get("count", 3),
            
            "image": {
                "repository": config.metadata.get("image", "inferloop/synthdata"),
                "pullPolicy": "IfNotPresent",
                "tag": config.metadata.get("version", "latest")
            },
            
            "nameOverride": "",
            "fullnameOverride": "",
            
            "serviceAccount": {
                "create": True,
                "annotations": {},
                "name": ""
            },
            
            "podAnnotations": {
                "prometheus.io/scrape": "true",
                "prometheus.io/port": "8080"
            },
            
            "podSecurityContext": {
                "fsGroup": 2000,
                "runAsNonRoot": True,
                "runAsUser": 1000
            },
            
            "securityContext": {
                "capabilities": {
                    "drop": ["ALL"]
                },
                "readOnlyRootFilesystem": True,
                "allowPrivilegeEscalation": False
            },
            
            "service": {
                "type": config.networking.get("service_type", "ClusterIP"),
                "port": 80,
                "targetPort": 8000,
                "annotations": {}
            },
            
            "ingress": {
                "enabled": config.networking.get("ingress_enabled", False),
                "className": "nginx",
                "annotations": {
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true"
                },
                "hosts": [{
                    "host": config.networking.get("hostname", "synthdata.local"),
                    "paths": [{
                        "path": "/",
                        "pathType": "Prefix"
                    }]
                }],
                "tls": [{
                    "secretName": f"{self.chart_name}-tls",
                    "hosts": [config.networking.get("hostname", "synthdata.local")]
                }]
            },
            
            "resources": {
                "limits": {
                    "cpu": str(int(config.compute.get("cpu", "2")) * 2),
                    "memory": str(int(config.compute.get("memory", "4Gi").rstrip("Gi")) * 2) + "Gi"
                },
                "requests": {
                    "cpu": config.compute.get("cpu", "2"),
                    "memory": config.compute.get("memory", "4Gi")
                }
            },
            
            "autoscaling": {
                "enabled": config.metadata.get("autoscaling_enabled", True),
                "minReplicas": config.metadata.get("min_replicas", 3),
                "maxReplicas": config.metadata.get("max_replicas", 10),
                "targetCPUUtilizationPercentage": 80,
                "targetMemoryUtilizationPercentage": 80
            },
            
            "persistence": {
                "enabled": bool(config.storage),
                "storageClass": config.storage.get("storage_class", "standard") if config.storage else "",
                "accessMode": "ReadWriteOnce",
                "size": config.storage.get("size", "100Gi") if config.storage else "100Gi",
                "existingClaim": ""
            },
            
            "postgresql": {
                "enabled": config.metadata.get("postgresql_enabled", True),
                "auth": {
                    "database": "synthdata",
                    "username": "synthdata",
                    "existingSecret": "postgres-credentials"
                },
                "primary": {
                    "persistence": {
                        "enabled": True,
                        "size": "100Gi"
                    }
                }
            },
            
            "minio": {
                "enabled": config.metadata.get("minio_enabled", True),
                "mode": "distributed",
                "replicas": 4,
                "persistence": {
                    "enabled": True,
                    "size": "500Gi"
                },
                "auth": {
                    "existingSecret": "minio-credentials"
                }
            },
            
            "monitoring": {
                "enabled": True,
                "serviceMonitor": {
                    "enabled": True,
                    "interval": "30s",
                    "path": "/metrics"
                }
            },
            
            "env": config.metadata.get("env", {}),
            
            "nodeSelector": {},
            "tolerations": [],
            "affinity": {
                "podAntiAffinity": {
                    "preferredDuringSchedulingIgnoredDuringExecution": [{
                        "weight": 100,
                        "podAffinityTerm": {
                            "labelSelector": {
                                "matchExpressions": [{
                                    "key": "app.kubernetes.io/name",
                                    "operator": "In",
                                    "values": [self.chart_name]
                                }]
                            },
                            "topologyKey": "kubernetes.io/hostname"
                        }
                    }]
                }
            }
        }
        
        # Add GPU support if specified
        if config.compute.get("gpu"):
            values["resources"]["limits"]["nvidia.com/gpu"] = str(config.compute["gpu"])
            values["nodeSelector"] = {"gpu": "nvidia"}
            values["tolerations"] = [{
                "key": "nvidia.com/gpu",
                "operator": "Exists",
                "effect": "NoSchedule"
            }]
        
        with open(self.chart_dir / "values.yaml", "w") as f:
            yaml.dump(values, f, default_flow_style=False, sort_keys=False)
    
    def _create_templates(self, config: ResourceConfig):
        """Create Kubernetes template files."""
        templates_dir = self.chart_dir / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        # Create deployment.yaml
        self._create_deployment_template(templates_dir)
        
        # Create service.yaml
        self._create_service_template(templates_dir)
        
        # Create ingress.yaml
        self._create_ingress_template(templates_dir)
        
        # Create hpa.yaml
        self._create_hpa_template(templates_dir)
        
        # Create pvc.yaml
        self._create_pvc_template(templates_dir)
        
        # Create serviceaccount.yaml
        self._create_serviceaccount_template(templates_dir)
        
        # Create configmap.yaml
        self._create_configmap_template(templates_dir)
        
        # Create servicemonitor.yaml for Prometheus
        self._create_servicemonitor_template(templates_dir)
    
    def _create_deployment_template(self, templates_dir: Path):
        """Create deployment template."""
        deployment = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "synthdata.fullname" . }}
  labels:
    {{- include "synthdata.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "synthdata.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
        {{- with .Values.podAnnotations }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
      labels:
        {{- include "synthdata.selectorLabels" . | nindent 8 }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "synthdata.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
        - name: {{ .Chart.Name }}
          securityContext:
            {{- toYaml .Values.securityContext | nindent 12 }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
            - name: metrics
              containerPort: 8080
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.postgresql.auth.existingSecret }}
                  key: database-url
            - name: S3_ENDPOINT
              value: "http://{{ .Release.Name }}-minio:9000"
            - name: S3_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.minio.auth.existingSecret }}
                  key: access-key
            - name: S3_SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: {{ .Values.minio.auth.existingSecret }}
                  key: secret-key
            {{- range $key, $value := .Values.env }}
            - name: {{ $key }}
              value: {{ $value | quote }}
            {{- end }}
          volumeMounts:
            - name: data
              mountPath: /data
            - name: config
              mountPath: /config
              readOnly: true
      volumes:
        - name: data
          {{- if .Values.persistence.enabled }}
          persistentVolumeClaim:
            claimName: {{ .Values.persistence.existingClaim | default (include "synthdata.fullname" .) }}
          {{- else }}
          emptyDir: {}
          {{- end }}
        - name: config
          configMap:
            name: {{ include "synthdata.fullname" . }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}"""
        
        with open(templates_dir / "deployment.yaml", "w") as f:
            f.write(deployment)
    
    def _create_service_template(self, templates_dir: Path):
        """Create service template."""
        service = """apiVersion: v1
kind: Service
metadata:
  name: {{ include "synthdata.fullname" . }}
  labels:
    {{- include "synthdata.labels" . | nindent 4 }}
  {{- with .Values.service.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: http
      protocol: TCP
      name: http
    - port: 8080
      targetPort: metrics
      protocol: TCP
      name: metrics
  selector:
    {{- include "synthdata.selectorLabels" . | nindent 4 }}"""
        
        with open(templates_dir / "service.yaml", "w") as f:
            f.write(service)
    
    def _create_ingress_template(self, templates_dir: Path):
        """Create ingress template."""
        ingress = """{{- if .Values.ingress.enabled -}}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "synthdata.fullname" . }}
  labels:
    {{- include "synthdata.labels" . | nindent 4 }}
  {{- with .Values.ingress.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  {{- if .Values.ingress.className }}
  ingressClassName: {{ .Values.ingress.className }}
  {{- end }}
  {{- if .Values.ingress.tls }}
  tls:
    {{- range .Values.ingress.tls }}
    - hosts:
        {{- range .hosts }}
        - {{ . | quote }}
        {{- end }}
      secretName: {{ .secretName }}
    {{- end }}
  {{- end }}
  rules:
    {{- range .Values.ingress.hosts }}
    - host: {{ .host | quote }}
      http:
        paths:
          {{- range .paths }}
          - path: {{ .path }}
            pathType: {{ .pathType }}
            backend:
              service:
                name: {{ include "synthdata.fullname" $ }}
                port:
                  number: {{ $.Values.service.port }}
          {{- end }}
    {{- end }}
{{- end }}"""
        
        with open(templates_dir / "ingress.yaml", "w") as f:
            f.write(ingress)
    
    def _create_hpa_template(self, templates_dir: Path):
        """Create horizontal pod autoscaler template."""
        hpa = """{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "synthdata.fullname" . }}
  labels:
    {{- include "synthdata.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "synthdata.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    {{- if .Values.autoscaling.targetCPUUtilizationPercentage }}
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
    {{- end }}
    {{- if .Values.autoscaling.targetMemoryUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}
    {{- end }}
{{- end }}"""
        
        with open(templates_dir / "hpa.yaml", "w") as f:
            f.write(hpa)
    
    def _create_pvc_template(self, templates_dir: Path):
        """Create persistent volume claim template."""
        pvc = """{{- if and .Values.persistence.enabled (not .Values.persistence.existingClaim) }}
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {{ include "synthdata.fullname" . }}
  labels:
    {{- include "synthdata.labels" . | nindent 4 }}
spec:
  accessModes:
    - {{ .Values.persistence.accessMode }}
  {{- if .Values.persistence.storageClass }}
  storageClassName: {{ .Values.persistence.storageClass }}
  {{- end }}
  resources:
    requests:
      storage: {{ .Values.persistence.size }}
{{- end }}"""
        
        with open(templates_dir / "pvc.yaml", "w") as f:
            f.write(pvc)
    
    def _create_serviceaccount_template(self, templates_dir: Path):
        """Create service account template."""
        sa = """{{- if .Values.serviceAccount.create -}}
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ include "synthdata.serviceAccountName" . }}
  labels:
    {{- include "synthdata.labels" . | nindent 4 }}
  {{- with .Values.serviceAccount.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
{{- end }}"""
        
        with open(templates_dir / "serviceaccount.yaml", "w") as f:
            f.write(sa)
    
    def _create_configmap_template(self, templates_dir: Path):
        """Create configmap template."""
        cm = """apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ include "synthdata.fullname" . }}
  labels:
    {{- include "synthdata.labels" . | nindent 4 }}
data:
  config.yaml: |
    server:
      port: 8000
      metrics_port: 8080
    database:
      connection_pool_size: 20
      max_connections: 100
    storage:
      bucket: synthdata
      prefix: data/
    features:
      privacy_preservation: true
      validation_enabled: true
      monitoring_enabled: true"""
        
        with open(templates_dir / "configmap.yaml", "w") as f:
            f.write(cm)
    
    def _create_servicemonitor_template(self, templates_dir: Path):
        """Create service monitor for Prometheus."""
        sm = """{{- if and .Values.monitoring.enabled .Values.monitoring.serviceMonitor.enabled }}
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {{ include "synthdata.fullname" . }}
  labels:
    {{- include "synthdata.labels" . | nindent 4 }}
spec:
  selector:
    matchLabels:
      {{- include "synthdata.selectorLabels" . | nindent 6 }}
  endpoints:
    - port: metrics
      interval: {{ .Values.monitoring.serviceMonitor.interval }}
      path: {{ .Values.monitoring.serviceMonitor.path }}
{{- end }}"""
        
        with open(templates_dir / "servicemonitor.yaml", "w") as f:
            f.write(sm)
    
    def _create_helpers(self):
        """Create _helpers.tpl file."""
        helpers = """{{/*
Expand the name of the chart.
*/}}
{{- define "synthdata.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "synthdata.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "synthdata.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "synthdata.labels" -}}
helm.sh/chart: {{ include "synthdata.chart" . }}
{{ include "synthdata.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "synthdata.selectorLabels" -}}
app.kubernetes.io/name: {{ include "synthdata.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "synthdata.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "synthdata.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}"""
        
        templates_dir = self.chart_dir / "templates"
        with open(templates_dir / "_helpers.tpl", "w") as f:
            f.write(helpers)
    
    def create_notes(self):
        """Create NOTES.txt for post-installation instructions."""
        notes = """1. Get the application URL by running these commands:
{{- if .Values.ingress.enabled }}
{{- range $host := .Values.ingress.hosts }}
  {{- range .paths }}
  http{{ if $.Values.ingress.tls }}s{{ end }}://{{ $host.host }}{{ .path }}
  {{- end }}
{{- end }}
{{- else if contains "NodePort" .Values.service.type }}
  export NODE_PORT=$(kubectl get --namespace {{ .Release.Namespace }} -o jsonpath="{.spec.ports[0].nodePort}" services {{ include "synthdata.fullname" . }})
  export NODE_IP=$(kubectl get nodes --namespace {{ .Release.Namespace }} -o jsonpath="{.items[0].status.addresses[0].address}")
  echo http://$NODE_IP:$NODE_PORT
{{- else if contains "LoadBalancer" .Values.service.type }}
     NOTE: It may take a few minutes for the LoadBalancer IP to be available.
           You can watch the status of by running 'kubectl get --namespace {{ .Release.Namespace }} svc -w {{ include "synthdata.fullname" . }}'
  export SERVICE_IP=$(kubectl get svc --namespace {{ .Release.Namespace }} {{ include "synthdata.fullname" . }} --template "{{"{{ range (index .status.loadBalancer.ingress 0) }}{{.}}{{ end }}"}}")
  echo http://$SERVICE_IP:{{ .Values.service.port }}
{{- else if contains "ClusterIP" .Values.service.type }}
  export POD_NAME=$(kubectl get pods --namespace {{ .Release.Namespace }} -l "app.kubernetes.io/name={{ include "synthdata.name" . }},app.kubernetes.io/instance={{ .Release.Name }}" -o jsonpath="{.items[0].metadata.name}")
  export CONTAINER_PORT=$(kubectl get pod --namespace {{ .Release.Namespace }} $POD_NAME -o jsonpath="{.spec.containers[0].ports[0].containerPort}")
  echo "Visit http://127.0.0.1:8080 to use your application"
  kubectl --namespace {{ .Release.Namespace }} port-forward $POD_NAME 8080:$CONTAINER_PORT
{{- end }}

2. Check the deployment status:
  kubectl get pods -l "app.kubernetes.io/name={{ include "synthdata.name" . }}" -n {{ .Release.Namespace }}

3. View application logs:
  kubectl logs -l "app.kubernetes.io/name={{ include "synthdata.name" . }}" -n {{ .Release.Namespace }}

4. Access Prometheus metrics:
  http://{{ include "synthdata.fullname" . }}.{{ .Release.Namespace }}.svc.cluster.local:8080/metrics"""
        
        templates_dir = self.chart_dir / "templates"
        with open(templates_dir / "NOTES.txt", "w") as f:
            f.write(notes)


class HelmDeployer:
    """Deploy applications using Helm."""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path
        self.helm_cmd = ["helm"]
        if kubeconfig_path:
            self.helm_cmd.extend(["--kubeconfig", kubeconfig_path])
    
    def install(
        self,
        release_name: str,
        chart_path: str,
        namespace: str = "default",
        values_file: Optional[str] = None,
        values: Optional[Dict[str, Any]] = None,
        wait: bool = True,
        timeout: str = "10m"
    ) -> DeploymentResult:
        """Install a Helm chart.
        
        Args:
            release_name: Name of the Helm release
            chart_path: Path to the chart directory
            namespace: Kubernetes namespace
            values_file: Path to values file
            values: Additional values to override
            wait: Wait for deployment to be ready
            timeout: Timeout for the operation
            
        Returns:
            DeploymentResult
        """
        cmd = self.helm_cmd + [
            "install", release_name, chart_path,
            "--namespace", namespace,
            "--create-namespace"
        ]
        
        if values_file:
            cmd.extend(["--values", values_file])
        
        if values:
            for key, value in values.items():
                cmd.extend(["--set", f"{key}={value}"])
        
        if wait:
            cmd.extend(["--wait", "--timeout", timeout])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            deployment_result = DeploymentResult()
            deployment_result.success = True
            deployment_result.message = f"Helm release '{release_name}' installed successfully"
            deployment_result.resource_id = f"{namespace}/{release_name}"
            deployment_result.metadata = {
                "release_name": release_name,
                "namespace": namespace,
                "chart": chart_path
            }
            
            return deployment_result
            
        except subprocess.CalledProcessError as e:
            deployment_result = DeploymentResult()
            deployment_result.success = False
            deployment_result.message = f"Helm install failed: {e.stderr}"
            return deployment_result
    
    def upgrade(
        self,
        release_name: str,
        chart_path: str,
        namespace: str = "default",
        values_file: Optional[str] = None,
        values: Optional[Dict[str, Any]] = None,
        wait: bool = True,
        timeout: str = "10m"
    ) -> DeploymentResult:
        """Upgrade a Helm release.
        
        Args:
            release_name: Name of the Helm release
            chart_path: Path to the chart directory
            namespace: Kubernetes namespace
            values_file: Path to values file
            values: Additional values to override
            wait: Wait for deployment to be ready
            timeout: Timeout for the operation
            
        Returns:
            DeploymentResult
        """
        cmd = self.helm_cmd + [
            "upgrade", release_name, chart_path,
            "--namespace", namespace,
            "--install"  # Install if doesn't exist
        ]
        
        if values_file:
            cmd.extend(["--values", values_file])
        
        if values:
            for key, value in values.items():
                cmd.extend(["--set", f"{key}={value}"])
        
        if wait:
            cmd.extend(["--wait", "--timeout", timeout])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            deployment_result = DeploymentResult()
            deployment_result.success = True
            deployment_result.message = f"Helm release '{release_name}' upgraded successfully"
            deployment_result.resource_id = f"{namespace}/{release_name}"
            
            return deployment_result
            
        except subprocess.CalledProcessError as e:
            deployment_result = DeploymentResult()
            deployment_result.success = False
            deployment_result.message = f"Helm upgrade failed: {e.stderr}"
            return deployment_result
    
    def uninstall(self, release_name: str, namespace: str = "default") -> bool:
        """Uninstall a Helm release.
        
        Args:
            release_name: Name of the Helm release
            namespace: Kubernetes namespace
            
        Returns:
            bool: True if successful
        """
        cmd = self.helm_cmd + [
            "uninstall", release_name,
            "--namespace", namespace
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def list_releases(self, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """List Helm releases.
        
        Args:
            namespace: Kubernetes namespace (all if None)
            
        Returns:
            List of release information
        """
        cmd = self.helm_cmd + ["list", "--output", "json"]
        
        if namespace:
            cmd.extend(["--namespace", namespace])
        else:
            cmd.append("--all-namespaces")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return []
    
    def get_values(self, release_name: str, namespace: str = "default") -> Dict[str, Any]:
        """Get values for a Helm release.
        
        Args:
            release_name: Name of the Helm release
            namespace: Kubernetes namespace
            
        Returns:
            Dict of values
        """
        cmd = self.helm_cmd + [
            "get", "values", release_name,
            "--namespace", namespace,
            "--output", "json"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return {}
    
    def rollback(
        self,
        release_name: str,
        revision: Optional[int] = None,
        namespace: str = "default",
        wait: bool = True
    ) -> bool:
        """Rollback a Helm release.
        
        Args:
            release_name: Name of the Helm release
            revision: Revision to rollback to (previous if None)
            namespace: Kubernetes namespace
            wait: Wait for rollback to complete
            
        Returns:
            bool: True if successful
        """
        cmd = self.helm_cmd + [
            "rollback", release_name,
            "--namespace", namespace
        ]
        
        if revision:
            cmd.append(str(revision))
        
        if wait:
            cmd.append("--wait")
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False