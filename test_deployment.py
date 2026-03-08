#!/usr/bin/env python3
"""Test your deployed NegotiateEnv on HuggingFace Spaces."""

import requests
import json
import time

ENV_URL = "https://kushaladhyaru-negotiate-env.hf.space"

def test_health():
    """Test health endpoint."""
    print("=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    try:
        response = requests.get(f"{ENV_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to connect to environment")
        print(f"   Error: {e}")
        print("\n⚠️  Make sure:")
        print("   1. You've pushed to HuggingFace Spaces")
        print("   2. The build has completed (check Logs tab)")
        print("   3. Space status shows 'Running'")
        return False

def test_reset():
    """Test reset endpoint."""
    print("\n" + "=" * 60)
    print("Testing Reset Endpoint")
    print("=" * 60)
    try:
        response = requests.post(f"{ENV_URL}/reset", json={}, timeout=10)
        if response.status_code == 200:
            obs = response.json()
            print("✅ Reset successful!")
            print(f"   Scenario: {obs.get('context', '')[:80]}...")
            print(f"   Your max price: ${obs.get('your_max_price')}")
            print(f"   Max turns: {obs.get('max_turns')}")
            print(f"   Turn number: {obs.get('turn_number')}")
            return True, obs
        else:
            print(f"❌ Reset failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False, None
    except Exception as e:
        print(f"❌ Reset test failed: {e}")
        return False, None

def test_step(obs):
    """Test step endpoint."""
    print("\n" + "=" * 60)
    print("Testing Step Endpoint")
    print("=" * 60)
    
    # Create a simple counter action
    action = {
        "action_type": "counter",
        "price_per_seat": obs.get('your_max_price', 100) * 0.9,
        "contract_length": 2.0,
        "annual_increase_cap": 5.0,
        "message": "Can we negotiate on the price?"
    }
    
    print(f"   Sending action: {action['action_type']}")
    print(f"   Price: ${action['price_per_seat']:.2f}")
    
    try:
        response = requests.post(f"{ENV_URL}/step", json=action, timeout=10)
        if response.status_code == 200:
            new_obs = response.json()
            print("✅ Step successful!")
            print(f"   AE response: {new_obs.get('ae_message', '')[:80]}...")
            print(f"   Done: {new_obs.get('done')}")
            print(f"   Reward: {new_obs.get('reward')}")
            print(f"   Turn: {new_obs.get('turn_number')}")
            return True
        else:
            print(f"❌ Step failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Step test failed: {e}")
        return False

def test_state():
    """Test state endpoint."""
    print("\n" + "=" * 60)
    print("Testing State Endpoint")
    print("=" * 60)
    try:
        response = requests.get(f"{ENV_URL}/state", timeout=10)
        if response.status_code == 200:
            state = response.json()
            print("✅ State retrieval successful!")
            print(f"   State: {json.dumps(state, indent=2)}")
            return True
        else:
            print(f"❌ State failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ State test failed: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("NegotiateEnv Deployment Test")
    print("=" * 60)
    print(f"Environment URL: {ENV_URL}")
    print()
    
    # Test health
    if not test_health():
        print("\n❌ Health check failed. Cannot proceed with other tests.")
        print("\nTroubleshooting:")
        print("1. Check Space status: https://huggingface.co/spaces/KushalAdhyaru/negotiate-env")
        print("2. View build logs in the 'Logs' tab")
        print("3. Wait for 'Running' status with green dot")
        return
    
    # Wait a moment
    time.sleep(1)
    
    # Test reset
    success, obs = test_reset()
    if not success:
        print("\n❌ Reset failed. Cannot proceed with step test.")
        return
    
    # Wait a moment
    time.sleep(1)
    
    # Test step
    test_step(obs)
    
    # Wait a moment
    time.sleep(1)
    
    # Test state
    test_state()
    
    print("\n" + "=" * 60)
    print("✅ All Tests Complete!")
    print("=" * 60)
    print("\nYour environment is ready for training!")
    print(f"\nUse this URL in your Colab notebook:")
    print(f"  ENV_URL = \"{ENV_URL}\"")
    print()

if __name__ == "__main__":
    main()
