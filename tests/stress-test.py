import asyncio
import httpx
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- KONFIGURACJA ---
parser = argparse.ArgumentParser()
parser.add_argument('--target_users', type=int, default=200, help='Docelowa liczba równoległych użytkowników')
parser.add_argument('--spawn_rate', type=int, default=2, help='Ilu użytkowników dodawać co sekundę')
parser.add_argument('--req_per_user', type=int, default=5, help='Ile zapytań wysyła każdy użytkownik')
args = parser.parse_args()

URL = "http://127.0.0.1:8002/matchCard"
# Upewnij się, że ta ścieżka istnieje
IMG_PATH = "tests/imgs/dragon_ball_fusion_world/810566.jpg" 

results = []

async def simulate_user(client, user_id, semaphore):
    """Symuluje jednego użytkownika."""
    async with semaphore:
        for _ in range(args.req_per_user):
            try:
                # Otwieramy plik asynchronicznie, aby nie blokować event loopa
                with open(IMG_PATH, 'rb') as f:
                    content = f.read()
                    files = {'file': (os.path.basename(IMG_PATH), content, 'image/jpeg')}
                    
                    start = time.perf_counter()
                    resp = await client.post(URL, files=files, timeout=None)
                    end = time.perf_counter()
                    
                    results.append({
                        'timestamp': start, 
                        'latency': end - start,
                        'status': resp.status_code
                    })
                
                if args.req_per_user > 1:
                    await asyncio.sleep(0.1) 
            except Exception as e:
                results.append({'timestamp': time.perf_counter(), 'latency': 0, 'status': f"err: {str(e)}"})

async def run_ramp_up():
    print(f"🚀 RAMP-UP START: {args.target_users} users, spawn: {args.spawn_rate}/s")
    test_start_time = time.perf_counter()
    
    # Limit połączeń w kliencie HTTP
    limits = httpx.Limits(max_connections=args.target_users, max_keepalive_connections=20)
    
    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        # Semafor kontrolujący tempo "wypuszczania" zadań do event loopa
        global_sem = asyncio.Semaphore(args.target_users)
        tasks = []
        
        for i in range(args.target_users):
            task = asyncio.create_task(simulate_user(client, i, global_sem))
            tasks.append(task)
            
            # Mechanizm Ramp-up
            if (i + 1) % args.spawn_rate == 0:
                print(f"  > Aktywnych użytkowników: {i + 1}...")
                await asyncio.sleep(1)
        
        await asyncio.gather(*tasks)

    # --- ANALIZA STATYSTYCZNA ---
    if not results:
        print("❌ Brak danych do analizy.")
        return

    results_df = pd.DataFrame(results).sort_values('timestamp')
    success_df = results_df[results_df['status'] == 200].copy()
    
    if success_df.empty:
        print(f"❌ Wszystkie zapytania zawiodły! Statusy: {results_df['status'].unique()}")
        return

    # Obliczenia
    test_duration = success_df['timestamp'].max() - success_df['timestamp'].min()
    rps = len(success_df) / test_duration if test_duration > 0 else 0
    latencies = success_df['latency'].values
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print(f"\n📊 STATYSTYKI KOŃCOWE (9800X3D)")
    print(f"{'='*30}")
    print(f"Średnia wydajność (RPS): {rps:.2f} req/s")
    print(f"Czas odpowiedzi (Avg):   {latencies.mean():.4f}s")
    print(f"Percentyl 95 (p95):     {p95:.4f}s")
    print(f"Percentyl 99 (p99):     {p99:.4f}s")
    print(f"Sukcesy:                {len(success_df)} / {len(results)}")
    print(f"{'='*30}")

    # --- GENEROWANIE WYKRESU ---
    plt.figure(figsize=(15, 8))
    
    # 1. Chmura punktów (Latency)
    relative_times = success_df['timestamp'] - test_start_time
    sc = plt.scatter(relative_times, success_df['latency'], 
                     alpha=0.5, c=success_df['latency'], cmap='viridis', s=30)
    
    # 2. Linia trendu (Średnia krocząca)
    success_df['rolling_avg'] = success_df['latency'].rolling(window=max(5, args.spawn_rate)).mean()
    plt.plot(relative_times, success_df['rolling_avg'], color='red', linewidth=2, label='Trend (Moving Avg)')

    # 3. Linie pomocnicze
    plt.axhline(y=p95, color='orange', linestyle='--', label=f'p95 ({p95:.2f}s)')
    plt.colorbar(sc, label='Latency (s)')
    
    plt.title(f"Ramp-up Test: {args.target_users} Users | RPS: {rps:.2f}")
    plt.xlabel("Sekundy od startu testu")
    plt.ylabel("Czas odpowiedzi (s)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    output_name = f"ramp_up_v3_{args.target_users}u.png"
    plt.savefig(output_name)
    print(f"✅ Wykres zapisany jako: {output_name}")
    plt.show()

if __name__ == "__main__":
    asyncio.run(run_ramp_up())