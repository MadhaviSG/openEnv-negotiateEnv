#!/bin/bash
# Deploy WebSocket fix and test all endpoints

echo "Step 1: Updating HF Space..."
cd hf_space
cp ../negotiate_env/server/app.py negotiate_env/server/app.py
git add negotiate_env/server/app.py
git commit -m "Enable WebSocket and fix session management"
git push
cd ..

echo ""
echo "Step 2: Waiting for Space to rebuild (90 seconds)..."
sleep 90

echo ""
echo "Step 3: Testing all endpoints..."
python3 test_all_endpoints.py

echo ""
echo "Step 4: Testing WebSocket..."
python3 - << 'EOF'
import asyncio
import websockets
import json

async def test_ws():
    url = "wss://kushaladhyaru-negotiate-env.hf.space"
    print(f"Testing WebSocket: {url}")
    print("="*60)
    
    try:
        async with websockets.connect(url, timeout=10) as ws:
            print("[OK] WebSocket connected!")
            
            # Reset
            await ws.send(json.dumps({"type": "reset"}))
            response = await ws.recv()
            data = json.loads(response)
            print("[OK] Reset successful!")
            
            # Step
            action = {
                "action_type": "probe",
                "price_per_seat": 0.0,
                "contract_length": 0.0,
                "annual_increase_cap": 0.0,
                "message": "test"
            }
            await ws.send(json.dumps({"type": "step", "action": action}))
            response = await ws.recv()
            data = json.loads(response)
            print("[OK] Step successful!")
            print(f"   Done: {data.get('observation', {}).get('done')}")
            print(f"   Reward: {data.get('observation', {}).get('reward')}")
            
            print("\n" + "="*60)
            print("[SUCCESS] WebSocket is fully working!")
            print("="*60)
            
    except Exception as e:
        print(f"\n[ERROR] WebSocket failed: {e}")
        print("\nYour Space may need more time to rebuild.")
        print("Wait another minute and run: python3 test_all_endpoints.py")

asyncio.run(test_ws())
EOF

echo ""
echo "Done! Check results above."
