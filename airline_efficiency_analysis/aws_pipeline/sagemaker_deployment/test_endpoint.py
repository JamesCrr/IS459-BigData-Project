"""
Quick Test Script for SageMaker Cascade Prediction Endpoint
============================================================

This script quickly tests your deployed SageMaker endpoint.

Usage:
    python test_endpoint.py --endpoint cascade-prediction-v2-endpoint
"""

import boto3
import json
import argparse
from datetime import datetime

def test_endpoint(endpoint_name, region='us-east-1'):
    """Test SageMaker endpoint with sample data"""
    
    print("="*80)
    print("TESTING SAGEMAKER CASCADE PREDICTION ENDPOINT")
    print("="*80)
    
    # Initialize client
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    # Test cases
    test_cases = [
        {
            'name': 'High Risk Flight (Incoming delay + tight turnaround)',
            'features': [
                # Temporal: 6PM, Wednesday, June
                18, 2, 6, 0, 1, 0, 0,
                # Flight: 800mi, 120min, not short-haul
                800, 120, 0,
                # Incoming delay: 25min delay from previous flight
                25, 1, 20,
                # Turnaround: Only 45min buffer (tight!)
                45, 1, 0, 1,
                # Utilization: 3rd flight of day
                3, 0, 1, 0,
                # Historical: Average route performance
                5.2, 12.3, 75.0, 8.5, 15.2, 6.8, 12.1
            ]
        },
        {
            'name': 'Normal Risk Flight (No incoming delay + good buffer)',
            'features': [
                # Temporal: 10AM, Tuesday, June
                10, 1, 6, 0, 0, 0, 0,
                # Flight: 500mi, 90min, short-haul
                500, 90, 1,
                # Incoming delay: No delay
                0, 0, 0,
                # Turnaround: Comfortable 2hr buffer
                120, 0, 0, 0,
                # Utilization: First flight
                1, 1, 1, 0,
                # Historical: Good route performance
                2.5, 8.0, 85.0, 5.0, 10.0, 4.0, 9.0
            ]
        },
        {
            'name': 'Critical Risk Flight (Late night + severe incoming delay)',
            'features': [
                # Temporal: 11PM, Friday, December (holiday season)
                23, 4, 12, 0, 0, 0, 1,
                # Flight: 1500mi, 180min, long-haul
                1500, 180, 0,
                # Incoming delay: Severe 45min delay
                45, 1, 40,
                # Turnaround: Critical 30min buffer only
                30, 1, 1, 1,
                # Utilization: 5th flight (late rotation)
                5, 0, 0, 1,
                # Historical: Poor route performance
                15.0, 20.0, 45.0, 12.0, 20.0, 14.0, 18.0
            ]
        }
    ]
    
    # Run tests
    results = []
    for idx, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {idx}: {test_case['name']}")
        print(f"{'='*80}")
        
        # Prepare CSV input
        csv_data = ','.join(map(str, test_case['features']))
        
        try:
            # Invoke endpoint
            print("\n📤 Sending prediction request...")
            start_time = datetime.now()
            
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='text/csv',
                Body=csv_data
            )
            
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds() * 1000
            
            # Parse response
            result = json.loads(response['Body'].read())
            prediction = result['predictions'][0]
            
            # Display results
            print(f"\n✅ PREDICTION SUCCESSFUL (latency: {latency:.0f}ms)")
            print(f"\n📊 Results:")
            print(f"   Cascade Probability: {prediction['cascade_probability']:.4f}")
            print(f"   Risk Tier: {prediction['risk_tier']}")
            print(f"   Cascade Prediction: {'YES' if prediction['cascade_prediction'] == 1 else 'NO'}")
            print(f"\n💡 Recommended Action:")
            print(f"   {prediction['recommended_action']}")
            
            results.append({
                'test_case': test_case['name'],
                'probability': prediction['cascade_probability'],
                'risk_tier': prediction['risk_tier'],
                'latency_ms': latency,
                'status': 'SUCCESS'
            })
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            results.append({
                'test_case': test_case['name'],
                'status': 'FAILED',
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\n✅ {success_count}/{len(test_cases)} tests passed")
    
    if success_count > 0:
        avg_latency = sum(r.get('latency_ms', 0) for r in results if r['status'] == 'SUCCESS') / success_count
        print(f"📈 Average latency: {avg_latency:.0f}ms")
        
        print(f"\n📊 Risk Distribution:")
        for tier in ['CRITICAL', 'HIGH', 'ELEVATED', 'NORMAL']:
            count = sum(1 for r in results if r.get('risk_tier') == tier)
            if count > 0:
                print(f"   {tier}: {count} flight(s)")
    
    # Detailed results
    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}")
    for result in results:
        print(f"\n{result['test_case']}:")
        if result['status'] == 'SUCCESS':
            print(f"  ✅ Probability: {result['probability']:.4f}")
            print(f"  ✅ Risk: {result['risk_tier']}")
            print(f"  ✅ Latency: {result['latency_ms']:.0f}ms")
        else:
            print(f"  ❌ Error: {result['error']}")
    
    print(f"\n{'='*80}")
    if success_count == len(test_cases):
        print("✅ ALL TESTS PASSED - ENDPOINT IS WORKING CORRECTLY!")
    else:
        print(f"⚠️  {len(test_cases) - success_count} TEST(S) FAILED")
    print(f"{'='*80}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test SageMaker cascade prediction endpoint')
    parser.add_argument('--endpoint', required=True, help='SageMaker endpoint name')
    parser.add_argument('--region', default='us-east-1', help='AWS region (default: us-east-1)')
    
    args = parser.parse_args()
    
    try:
        results = test_endpoint(args.endpoint, args.region)
        
        # Exit code based on results
        success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
        exit(0 if success_count == len(results) else 1)
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
