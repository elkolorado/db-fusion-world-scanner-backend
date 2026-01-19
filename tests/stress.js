import http from 'k6/http';
import { check, sleep } from 'k6';

const binFile = open('./imgs/riftbound/845712.jpg', 'b');

export const options = {
    scenarios: {
        load_500_rps: {
            executor: 'constant-arrival-rate',
            rate: 500,              // 🎯 500 requestów / sekundę
            timeUnit: '1s',
            duration: '30s',

            // k6 będzie skalował VU do tej liczby
            preAllocatedVUs: 100,
            maxVUs: 300,
        },
    },
    thresholds: {
        http_req_duration: [
            'p(50)<400',
            'p(95)<1000',
        ],
        http_req_failed: ['rate<0.01'],
    },
};

export default function () {
    const url = 'http://localhost:8002/matchCard?tcg_name=riftbound';

    // 🧠 mikro-jitter (0–50 ms) — symuluje scheduling, GC, sieć
    sleep(Math.random() * 0.05);

    const data = {
        file: http.file(binFile, 'card.jpg', 'image/jpeg'),
    };

    const res = http.post(url, data);

    check(res, {
        'status 200': (r) => r.status === 200,
    });
}
