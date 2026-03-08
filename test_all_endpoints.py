#!/usr/bin/env python3
"""Test all endpoints of the NegotiateEnv environment."""

import requests
import json

ENV_URL = "https://kushaladhyaru-negotiate-env.hf.space"

def test_endpoint(method, endpoint, data=None, description=""):
    """Test a single endpoint."""
    url = f"{ENV_URL}{endpoint}"
    print(f"\n{'='*60}")
    print(f"Testing: {method} {endpoint}")
    if description:
        print(f"Description: {description}")
    print('='*60)
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data or {}, timeout=10)
        else:
            print(f"Unknown method: {method}")
            return False
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Response (preview):")
                print(json.dumps(result, indent=2)[:500])
                if len(json.dumps(result)) > 500:
                    print("... (truncated)")
                print("\n✓ SUCCESS")
                return True
            except:
                print(f"Response Text: {response.text[:200]}")
                print("\n✓ SUCCESS (non-JSON)")
                return True
        else:
            print(f"Error: {response.text[:200]}")
            print("\n✗ FAILED")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        print("\n✗ FAILED")
        return False


def main():
    print(f"\n{'#'*60}")
    print(f"# Testing NegotiateEnv Environment")
    print(f"# URL: {ENV_URL}")
    print(f"{'#'*60}")
    
    results = {}
    
    # Test 1: Health check
    results['health'] = test_endpoint(
        "GET", "/health",
        description="Check if service is running"
    )
    
    # Test 2: Environment info
    results['info'] = test_endpoint(
        "GET", "/info",
        description="Get environment metadata"
    )
    
    # Test 3: Action schema
    results['action_schema'] = test_endpoint(
        "GET", "/action_schema",
        description="Get JSON schema for actions"
    )
    
    # Test 4: Observation schema
    results['observation_schema'] = test_endpoint(
        "GET", "/observation_schema",
        description="Get JSON schema for observations"
    )
    
    # Test 5: Reset (start new episode)
    results['reset'] = test_endpoint(
        "POST", "/reset",
        data={},
        description="Start a new negotiation episode"
    )
    
    # Test 6: Step (take action)
    action = {
        "action_type": "probe",
        "price_per_seat": 0.0,
        "contract_length": 0.0,
        "annual_increase_cap": 0.0,
        "message": "Can you tell me more about your pricing?"
    }
    results['step'] = test_endpoint(
        "POST", "/step",
        data={"action": action},
        description="Take a probe action"
    )
    
    # Test 7: State (get current state)
    results['state'] = test_endpoint(
        "GET", "/state",
        description="Get current environment state"
    )
    
    # Summary
    print(f"\n\n{'#'*60}")
    print(f"# TEST SUMMARY")
    print(f"{'#'*60}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for endpoint, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} - {endpoint}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All endpoints are working!")
        print("✓ Environment is ready for training!")
    else:
        print("\n✗ Some endpoints failed")
        print("✗ Check your HF Space logs")
    
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
