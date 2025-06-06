# audio_synth/monitoring/dashboard.py
"""
Real-time monitoring dashboard
"""

from flask import Flask, render_template, jsonify, request
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class MonitoringDashboard:
    """Real-time monitoring dashboard for audio synthesis"""
    
    def __init__(self, metrics_collector, alert_manager, port: int = 5000):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self._setup_routes()
        
        logger.info(f"Monitoring dashboard initialized on port {port}")
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/stats')
        def get_stats():
            """Get current statistics"""
            return jsonify(self.metrics_collector.get_current_stats())
        
        @self.app.route('/api/trends')
        def get_trends():
            """Get performance trends"""
            hours = request.args.get('hours', 24, type=int)
            return jsonify(self.metrics_collector.get_performance_trends(hours))
        
        @self.app.route('/api/quality')
        def get_quality():
            """Get quality analysis"""
            return jsonify(self.metrics_collector.get_quality_analysis())
        
        @self.app.route('/api/alerts')
        def get_alerts():
            """Get active alerts"""
            return jsonify({
                "active": self.alert_manager.get_active_alerts(),
                "history": self.alert_manager.get_alert_history()
            })
        
        @self.app.route('/api/alerts/<alert_name>/resolve', methods=['POST'])
        def resolve_alert(alert_name):
            """Resolve an alert"""
            self.alert_manager.resolve_alert(alert_name)
            return jsonify({"status": "resolved"})
        
        @self.app.route('/api/export')
        def export_metrics():
            """Export metrics"""
            format_type = request.args.get('format', 'json')
            
            try:
                data = self.metrics_collector.export_metrics(format_type)
                
                if format_type == 'json':
                    return jsonify(json.loads(data))
                else:
                    return data, 200, {'Content-Type': 'text/plain'}
                    
            except Exception as e:
                return jsonify({"error": str(e)}), 400
        
        @self.app.route('/health')
        def health():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "timestamp": time.time(),
                "metrics_collector": "active",
                "alert_manager": "active"
            })
    
    def run(self, debug: bool = False):
        """Run the dashboard"""
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)

# Dashboard HTML template would be saved as templates/dashboard.html
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Synthesis Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .chart-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .alerts-section {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .alert {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 4px solid;
        }
        .alert.error {
            background-color: #ffeaea;
            border-color: #e74c3c;
        }
        .alert.warning {
            background-color: #fff3cd;
            border-color: #f39c12;
        }
        .alert.info {
            background-color: #d1ecf1;
            border-color: #17a2b8;
        }
        .resolve-btn {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            float: right;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-good { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-error { background-color: #dc3545; }
        .refresh-info {
            text-align: center;
            color: #666;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎵 Audio Synthesis Monitoring Dashboard</h1>
        <p>Real-time performance and quality monitoring</p>
        <div id="lastUpdate" class="refresh-info">Loading...</div>
    </div>

    <div class="stats-grid" id="statsGrid">
        <!-- Stats cards will be populated here -->
    </div>

    <div class="chart-container">
        <div class="chart-title">Request Volume & Success Rate</div>
        <canvas id="requestsChart" height="100"></canvas>
    </div>

    <div class="chart-container">
        <div class="chart-title">Response Time Trends</div>
        <canvas id="responseTimeChart" height="100"></canvas>
    </div>

    <div class="chart-container">
        <div class="chart-title">Method Performance Comparison</div>
        <canvas id="methodsChart" height="100"></canvas>
    </div>

    <div class="alerts-section">
        <h2>🚨 Active Alerts</h2>
        <div id="alertsList">
            <!-- Alerts will be populated here -->
        </div>
    </div>

    <script>
        // Initialize charts
        let requestsChart, responseTimeChart, methodsChart;
        
        function initializeCharts() {
            // Requests chart
            const ctx1 = document.getElementById('requestsChart').getContext('2d');
            requestsChart = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Requests',
                        data: [],
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        yAxisID: 'y'
                    }, {
                        label: 'Success Rate (%)',
                        data: [],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            min: 0,
                            max: 100,
                            grid: {
                                drawOnChartArea: false,
                            },
                        }
                    }
                }
            });

            // Response time chart
            const ctx2 = document.getElementById('responseTimeChart').getContext('2d');
            responseTimeChart = new Chart(ctx2, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Avg Response Time (s)',
                        data: [],
                        borderColor: '#f39c12',
                        backgroundColor: 'rgba(243, 156, 18, 0.1)'
                    }, {
                        label: 'P95 Response Time (s)',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            // Methods chart
            const ctx3 = document.getElementById('methodsChart').getContext('2d');
            methodsChart = new Chart(ctx3, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Requests',
                        data: [],
                        backgroundColor: ['#667eea', '#764ba2', '#f093fb', '#f5576c']
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }

        function updateStats(stats) {
            const statsGrid = document.getElementById('statsGrid');
            const successRate = (stats.success_rate * 100).toFixed(1);
            const statusClass = stats.success_rate > 0.95 ? 'status-good' : 
                              stats.success_rate > 0.8 ? 'status-warning' : 'status-error';
            
            statsGrid.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${stats.requests_last_hour}</div>
                    <div class="stat-label">Requests (Last Hour)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">
                        <span class="status-indicator ${statusClass}"></span>
                        ${successRate}%
                    </div>
                    <div class="stat-label">Success Rate</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.avg_duration.toFixed(2)}s</div>
                    <div class="stat-label">Avg Response Time</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.total_samples}</div>
                    <div class="stat-label">Samples Generated</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.active_requests}</div>
                    <div class="stat-label">Active Requests</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.system?.memory_percent?.toFixed(1) || 'N/A'}%</div>
                    <div class="stat-label">Memory Usage</div>
                </div>
            `;
        }

        function updateTrends(trends) {
            if (!trends.hourly_trends || trends.hourly_trends.length === 0) return;

            const labels = trends.hourly_trends.map(t => new Date(t.timestamp * 1000).toLocaleTimeString());
            const requests = trends.hourly_trends.map(t => t.requests);
            const successRates = trends.hourly_trends.map(t => t.success_rate * 100);
            const avgTimes = trends.hourly_trends.map(t => t.avg_duration);
            const p95Times = trends.hourly_trends.map(t => t.p95_duration);

            // Update requests chart
            requestsChart.data.labels = labels;
            requestsChart.data.datasets[0].data = requests;
            requestsChart.data.datasets[1].data = successRates;
            requestsChart.update();

            // Update response time chart
            responseTimeChart.data.labels = labels;
            responseTimeChart.data.datasets[0].data = avgTimes;
            responseTimeChart.data.datasets[1].data = p95Times;
            responseTimeChart.update();
        }

        function updateMethods(stats) {
            if (!stats.method_breakdown) return;

            const methods = Object.keys(stats.method_breakdown);
            const counts = methods.map(m => stats.method_breakdown[m].count);

            methodsChart.data.labels = methods.map(m => m.toUpperCase());
            methodsChart.data.datasets[0].data = counts;
            methodsChart.update();
        }

        function updateAlerts(alertsData) {
            const alertsList = document.getElementById('alertsList');
            const activeAlerts = alertsData.active || [];

            if (activeAlerts.length === 0) {
                alertsList.innerHTML = '<p style="color: #28a745;">✅ No active alerts</p>';
                return;
            }

            alertsList.innerHTML = activeAlerts.map(alert => `
                <div class="alert ${alert.severity}">
                    <strong>${alert.name}</strong> - ${alert.message}
                    <button class="resolve-btn" onclick="resolveAlert('${alert.name}')">Resolve</button>
                    <br>
                    <small>${new Date(alert.timestamp * 1000).toLocaleString()}</small>
                </div>
            `).join('');
        }

        function resolveAlert(alertName) {
            axios.post(`/api/alerts/${alertName}/resolve`)
                .then(() => {
                    loadAlerts();
                })
                .catch(err => {
                    console.error('Failed to resolve alert:', err);
                });
        }

        async function loadData() {
            try {
                const [stats, trends, alerts] = await Promise.all([
                    axios.get('/api/stats'),
                    axios.get('/api/trends?hours=24'),
                    axios.get('/api/alerts')
                ]);

                updateStats(stats.data);
                updateTrends(trends.data);
                updateMethods(stats.data);
                updateAlerts(alerts.data);

                document.getElementById('lastUpdate').textContent = 
                    `Last updated: ${new Date().toLocaleTimeString()}`;

            } catch (error) {
                console.error('Failed to load data:', error);
                document.getElementById('lastUpdate').textContent = 
                    `⚠️ Error loading data: ${error.message}`;
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
            loadData();
            
            // Refresh every 30 seconds
            setInterval(loadData, 30000);
        });
    </script>
</body>
</html>
'''