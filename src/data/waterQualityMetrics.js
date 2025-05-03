class WaterQualityMetrics {
  constructor() {
    this.metrics = {
      temperature: null,
      pH: null,
      turbidity: null,
      dissolvedOxygen: null,
      conductivity: null,
      timestamp: null,
    };
    this.subscribers = new Set();
  }

  updateMetrics(data) {
    try {
      // Parse the incoming data string
      const parsedData = JSON.parse(data);

      // Update metrics with new data
      this.metrics = {
        ...this.metrics,
        ...parsedData,
        timestamp: new Date().toISOString(),
      };

      // Notify all subscribers
      this.notifySubscribers();
    } catch (error) {
      console.error("Error parsing sensor data:", error);
    }
  }

  subscribe(callback) {
    this.subscribers.add(callback);
    // Immediately send current metrics to new subscriber
    callback(this.metrics);
  }

  unsubscribe(callback) {
    this.subscribers.delete(callback);
  }

  notifySubscribers() {
    this.subscribers.forEach((callback) => callback(this.metrics));
  }

  getMetrics() {
    return this.metrics;
  }

  getMetric(metricName) {
    return this.metrics[metricName];
  }
}

// Create a singleton instance
const waterQualityMetrics = new WaterQualityMetrics();

export default waterQualityMetrics;
