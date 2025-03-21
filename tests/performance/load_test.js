import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '10s', target: 10 }, // Ramp up to 10 users
    { duration: '10s', target: 10 }, // Stay at 10 users
    { duration: '10s', target: 0 },  // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests should be below 500ms
    http_req_failed: ['rate<0.01'],   // Less than 1% of requests should fail
  },
};

const BASE_URL = __ENV.TARGET_URL || 'http://localhost:8000';

const SAMPLE_PAYLOAD = {
  passenger_id: 1,
  pclass: 3,
  name: "Test, Mr. Performance",
  sex: "male",
  age: 22.0,
  sibsp: 1,
  parch: 0,
  ticket: "TEST123",
  fare: 7.25,
  cabin: "",
  embarked: "S"
};

export default function () {
  // Health check
  const healthCheck = http.get(`${BASE_URL}/health`);
  check(healthCheck, {
    'health check status is 200': (r) => r.status === 200,
  });

  // Metrics check
  const metricsCheck = http.get(`${BASE_URL}/metrics`);
  check(metricsCheck, {
    'metrics check status is 200': (r) => r.status === 200,
  });

  // Prediction request
  const predictionRes = http.post(
    `${BASE_URL}/api/v1/predict`,
    JSON.stringify(SAMPLE_PAYLOAD),
    {
      headers: { 'Content-Type': 'application/json' },
    }
  );

  check(predictionRes, {
    'prediction status is 200': (r) => r.status === 200,
    'prediction has correct structure': (r) => {
      const body = JSON.parse(r.body);
      return body.hasOwnProperty('prediction');
    },
  });

  sleep(1);
} 