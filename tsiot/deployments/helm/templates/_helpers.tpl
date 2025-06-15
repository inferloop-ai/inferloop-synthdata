{{/*
Expand the name of the chart.
*/}}
{{- define "tsiot.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "tsiot.fullname" -}}
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
{{- define "tsiot.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "tsiot.labels" -}}
helm.sh/chart: {{ include "tsiot.chart" . }}
{{ include "tsiot.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- with .Values.global.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "tsiot.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tsiot.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
TSIoT Server labels
*/}}
{{- define "tsiot.server.labels" -}}
helm.sh/chart: {{ include "tsiot.chart" . }}
{{ include "tsiot.server.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: server
{{- with .Values.global.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
TSIoT Server selector labels
*/}}
{{- define "tsiot.server.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tsiot.name" . }}-server
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
TSIoT Worker labels
*/}}
{{- define "tsiot.worker.labels" -}}
helm.sh/chart: {{ include "tsiot.chart" . }}
{{ include "tsiot.worker.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: worker
{{- with .Values.global.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end }}

{{/*
TSIoT Worker selector labels
*/}}
{{- define "tsiot.worker.selectorLabels" -}}
app.kubernetes.io/name: {{ include "tsiot.name" . }}-worker
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use for TSIoT Server
*/}}
{{- define "tsiot.server.serviceAccountName" -}}
{{- if .Values.security.rbac.serviceAccounts.server.create }}
{{- default (printf "%s-server" (include "tsiot.fullname" .)) .Values.security.rbac.serviceAccounts.server.name }}
{{- else }}
{{- default "default" .Values.security.rbac.serviceAccounts.server.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use for TSIoT Worker
*/}}
{{- define "tsiot.worker.serviceAccountName" -}}
{{- if .Values.security.rbac.serviceAccounts.worker.create }}
{{- default (printf "%s-worker" (include "tsiot.fullname" .)) .Values.security.rbac.serviceAccounts.worker.name }}
{{- else }}
{{- default "default" .Values.security.rbac.serviceAccounts.worker.name }}
{{- end }}
{{- end }}

{{/*
Create a default fully qualified server name.
*/}}
{{- define "tsiot.server.fullname" -}}
{{- printf "%s-server" (include "tsiot.fullname" .) }}
{{- end }}

{{/*
Create a default fully qualified worker name.
*/}}
{{- define "tsiot.worker.fullname" -}}
{{- printf "%s-worker" (include "tsiot.fullname" .) }}
{{- end }}

{{/*
Create the server image name
*/}}
{{- define "tsiot.server.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" }}
{{- $repository := .Values.server.image.repository }}
{{- $tag := .Values.server.image.tag | default .Chart.AppVersion }}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/*
Create the worker image name
*/}}
{{- define "tsiot.worker.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" }}
{{- $repository := .Values.worker.image.repository }}
{{- $tag := .Values.worker.image.tag | default .Chart.AppVersion }}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/*
Create the CLI image name
*/}}
{{- define "tsiot.cli.image" -}}
{{- $registry := .Values.global.imageRegistry | default "" }}
{{- $repository := .Values.cli.image.repository }}
{{- $tag := .Values.cli.image.tag | default .Chart.AppVersion }}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- else }}
{{- printf "%s:%s" $repository $tag }}
{{- end }}
{{- end }}

{{/*
Create a secret name for database credentials
*/}}
{{- define "tsiot.database.secretName" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" .Release.Name }}
{{- else if .Values.postgresql.external.existingSecret }}
{{- .Values.postgresql.external.existingSecret }}
{{- else }}
{{- printf "%s-database-credentials" (include "tsiot.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Create a secret name for Redis credentials
*/}}
{{- define "tsiot.redis.secretName" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis" .Release.Name }}
{{- else if .Values.redis.external.existingSecret }}
{{- .Values.redis.external.existingSecret }}
{{- else }}
{{- printf "%s-redis-credentials" (include "tsiot.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Create a secret name for Kafka credentials
*/}}
{{- define "tsiot.kafka.secretName" -}}
{{- if .Values.kafka.enabled }}
{{- printf "%s-kafka" .Release.Name }}
{{- else if .Values.kafka.external.existingSecret }}
{{- .Values.kafka.external.existingSecret }}
{{- else }}
{{- printf "%s-kafka-credentials" (include "tsiot.fullname" .) }}
{{- end }}
{{- end }}

{{/*
Create database host
*/}}
{{- define "tsiot.database.host" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" .Release.Name }}
{{- else }}
{{- .Values.postgresql.external.host }}
{{- end }}
{{- end }}

{{/*
Create database port
*/}}
{{- define "tsiot.database.port" -}}
{{- if .Values.postgresql.enabled }}
{{- "5432" }}
{{- else }}
{{- .Values.postgresql.external.port | toString }}
{{- end }}
{{- end }}

{{/*
Create Redis host
*/}}
{{- define "tsiot.redis.host" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis-master" .Release.Name }}
{{- else }}
{{- .Values.redis.external.host }}
{{- end }}
{{- end }}

{{/*
Create Redis port
*/}}
{{- define "tsiot.redis.port" -}}
{{- if .Values.redis.enabled }}
{{- "6379" }}
{{- else }}
{{- .Values.redis.external.port | toString }}
{{- end }}
{{- end }}

{{/*
Create Kafka brokers
*/}}
{{- define "tsiot.kafka.brokers" -}}
{{- if .Values.kafka.enabled }}
{{- printf "%s-kafka:9092" .Release.Name }}
{{- else }}
{{- join "," .Values.kafka.external.brokers }}
{{- end }}
{{- end }}

{{/*
Create InfluxDB URL
*/}}
{{- define "tsiot.influxdb.url" -}}
{{- if .Values.influxdb.enabled }}
{{- printf "http://%s-influxdb:8086" .Release.Name }}
{{- else }}
{{- .Values.influxdb.external.url }}
{{- end }}
{{- end }}

{{/*
Create Elasticsearch URL
*/}}
{{- define "tsiot.elasticsearch.url" -}}
{{- if .Values.elasticsearch.enabled }}
{{- printf "http://%s-elasticsearch-master:9200" .Release.Name }}
{{- else }}
{{- .Values.elasticsearch.external.url }}
{{- end }}
{{- end }}

{{/*
Security context for containers
*/}}
{{- define "tsiot.securityContext" -}}
{{- with .Values.global.securityContext }}
securityContext:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Pod security context
*/}}
{{- define "tsiot.podSecurityContext" -}}
{{- with .Values.global.securityContext }}
securityContext:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Image pull secrets
*/}}
{{- define "tsiot.imagePullSecrets" -}}
{{- if .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.global.imagePullSecrets }}
  - name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Validate required values
*/}}
{{- define "tsiot.validateValues" -}}
{{- if and (not .Values.postgresql.enabled) (not .Values.postgresql.external.host) }}
{{- fail "Either postgresql.enabled must be true or postgresql.external.host must be specified" }}
{{- end }}
{{- if and (not .Values.redis.enabled) (not .Values.redis.external.host) }}
{{- fail "Either redis.enabled must be true or redis.external.host must be specified" }}
{{- end }}
{{- end }}

{{/*
Environment variables for database connection
*/}}
{{- define "tsiot.database.env" -}}
- name: TSIOT_DB_HOST
  value: {{ include "tsiot.database.host" . | quote }}
- name: TSIOT_DB_PORT
  value: {{ include "tsiot.database.port" . | quote }}
- name: TSIOT_DB_NAME
  value: {{ .Values.postgresql.auth.database | default "tsiot" | quote }}
- name: TSIOT_DB_USER
  valueFrom:
    secretKeyRef:
      name: {{ include "tsiot.database.secretName" . }}
      key: {{ .Values.postgresql.external.userKey | default "username" }}
- name: TSIOT_DB_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "tsiot.database.secretName" . }}
      key: {{ .Values.postgresql.external.passwordKey | default "password" }}
{{- end }}

{{/*
Environment variables for Redis connection
*/}}
{{- define "tsiot.redis.env" -}}
- name: TSIOT_REDIS_HOST
  value: {{ include "tsiot.redis.host" . | quote }}
- name: TSIOT_REDIS_PORT
  value: {{ include "tsiot.redis.port" . | quote }}
{{- if or .Values.redis.auth.enabled .Values.redis.external.existingSecret }}
- name: TSIOT_REDIS_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "tsiot.redis.secretName" . }}
      key: {{ .Values.redis.external.passwordKey | default "redis-password" }}
{{- end }}
{{- end }}

{{/*
Environment variables for Kafka connection
*/}}
{{- define "tsiot.kafka.env" -}}
- name: TSIOT_KAFKA_BROKERS
  value: {{ include "tsiot.kafka.brokers" . | quote }}
{{- if .Values.kafka.external.existingSecret }}
- name: TSIOT_KAFKA_USERNAME
  valueFrom:
    secretKeyRef:
      name: {{ include "tsiot.kafka.secretName" . }}
      key: {{ .Values.kafka.external.usernameKey | default "username" }}
- name: TSIOT_KAFKA_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "tsiot.kafka.secretName" . }}
      key: {{ .Values.kafka.external.passwordKey | default "password" }}
{{- end }}
{{- end }}

{{/*
Common environment variables
*/}}
{{- define "tsiot.common.env" -}}
- name: TSIOT_ENV
  value: {{ .Values.global.environment | quote }}
- name: TZ
  value: {{ .Values.global.timezone | quote }}
- name: TSIOT_NAMESPACE
  valueFrom:
    fieldRef:
      fieldPath: metadata.namespace
- name: TSIOT_POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: TSIOT_POD_IP
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
- name: TSIOT_NODE_NAME
  valueFrom:
    fieldRef:
      fieldPath: spec.nodeName
{{- end }}

{{/*
Storage class
*/}}
{{- define "tsiot.storageClass" -}}
{{- if .Values.global.storageClass }}
storageClassName: {{ .Values.global.storageClass | quote }}
{{- end }}
{{- end }}

{{/*
Monitoring annotations
*/}}
{{- define "tsiot.monitoring.annotations" -}}
prometheus.io/scrape: "true"
prometheus.io/port: {{ .port | quote }}
prometheus.io/path: {{ .path | default "/metrics" | quote }}
{{- end }}