from prometheus_client import Counter, Gauge, REGISTRY


def get_or_create_metric(name, description, metric_type="counter", labelnames="None"):
    registry = REGISTRY._names_to_collectors
    labelnames = labelnames or []
    if name in registry:
        return registry[name]
    if metric_type == "counter":
        return Counter(name, description)
    elif metric_type == "gauge":
        if labelnames == "None":
            return Gauge(name, description)
        else:
            return Gauge(name, description, labelnames=labelnames)
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")


health_requests = get_or_create_metric(
    "health_requests_total", "Total health check requests"
)
websocket_url_requests = get_or_create_metric(
    "websocket_url_requests_total", "Total WebSocket URL requests"
)
active_websockets = get_or_create_metric(
    "active_websockets", "Number of active WebSocket connections", "gauge"
)
visual_thoughts_reconstructed = get_or_create_metric(
    "visual_thoughts_reconstructed_total", "Total visual-thoughts processed"
)
websocket_errors = get_or_create_metric(
    "websocket_errors_total", "Total WebSocket errors"
)


class Metrics:
    def __init__(self):
        self.health_requests = health_requests
        self.websocket_url_requests = websocket_url_requests
        self.active_websockets = active_websockets
        self.visual_thoughts_reconstructed = visual_thoughts_reconstructed
        self.websocket_errors = websocket_errors


metrics = Metrics()

///

from prometheus_client import Counter, Gauge, REGISTRY


def get_or_create_metric(name, description, metric_type="counter", labelnames="None"):
    registry = REGISTRY._names_to_collectors
    labelnames = labelnames or []
    if name in registry:
        return registry[name]
    if metric_type == "counter":
        return Counter(name, description)
    elif metric_type == "gauge":
        if labelnames == "None":
            return Gauge(name, description)
        else:
            return Gauge(name, description, labelnames=labelnames)
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")


health_requests = get_or_create_metric(
    "health_requests_total", "Total health check requests"
)
websocket_url_requests = get_or_create_metric(
    "websocket_url_requests_total", "Total WebSocket URL requests"
)
active_websockets = get_or_create_metric(
    "active_websockets", "Number of active WebSocket connections", "gauge"
)
visual_thoughts_simulated = get_or_create_metric(
    "visual_thoughts_simulated_total", "Total visual-thoughts processed"
)
websocket_errors = get_or_create_metric(
    "websocket_errors_total", "Total WebSocket errors"
)


class Metrics:
    def __init__(self):
        self.health_requests = health_requests
        self.websocket_url_requests = websocket_url_requests
        self.active_websockets = active_websockets
        self.visual_thoughts_simulated = visual_thoughts_simulated
        self.websocket_errors = websocket_errors


metrics = Metrics()
