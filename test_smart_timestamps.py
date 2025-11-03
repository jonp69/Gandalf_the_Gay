#!/usr/bin/env python3
"""
Smart Timestamp Parser Test Script

This script demonstrates and tests the smart timestamp parsing functionality
that automatically detects different time formats.
"""

import re
import sys


def parse_smart_timestamp(timestamp_str: str) -> float:
    """
    Smart timestamp parser that handles multiple formats automatically.
    """
    timestamp_str = str(timestamp_str).strip()
    
    # Handle pure numeric values (seconds)
    try:
        return float(timestamp_str)
    except ValueError:
        pass
    
    # Handle HH:MM:SS or MM:SS formats
    if ':' in timestamp_str:
        parts = timestamp_str.split(':')
        
        if len(parts) == 2:  # MM:SS
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
            
        elif len(parts) == 3:  # HH:MM:SS
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    
    # Handle text formats like "1h23m45.5s", "2m15s", "45s"
    hour_match = re.search(r'(\d+(?:\.\d+)?)h', timestamp_str.lower())
    min_match = re.search(r'(\d+(?:\.\d+)?)m', timestamp_str.lower())
    sec_match = re.search(r'(\d+(?:\.\d+)?)s', timestamp_str.lower())
    
    total_seconds = 0.0
    
    if hour_match:
        total_seconds += float(hour_match.group(1)) * 3600
    if min_match:
        total_seconds += float(min_match.group(1)) * 60
    if sec_match:
        total_seconds += float(sec_match.group(1))
    
    if total_seconds > 0:
        return total_seconds
    
    raise ValueError(f"Could not parse timestamp '{timestamp_str}'")


def format_seconds(seconds: float) -> str:
    """Convert seconds back to HH:MM:SS.s format for display."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:06.3f}"
    else:
        return f"{minutes}:{secs:06.3f}"


def test_timestamp_formats():
    """Test various timestamp formats."""
    
    print("🕐 Smart Timestamp Parser Test")
    print("=" * 50)
    
    # Test cases: (input, expected_seconds, description)
    test_cases = [
        # Seconds only
        ("45.5", 45.5, "Decimal seconds"),
        ("123", 123.0, "Integer seconds"),
        ("0.5", 0.5, "Fractional seconds"),
        
        # MM:SS format
        ("1:23", 83.0, "Minutes:Seconds (integer)"),
        ("1:23.5", 83.5, "Minutes:Seconds (decimal)"),
        ("0:45.2", 45.2, "Zero minutes"),
        ("10:00", 600.0, "Exact minutes"),
        
        # HH:MM:SS format
        ("1:02:30", 3750.0, "Hours:Minutes:Seconds"),
        ("0:05:45.5", 345.5, "Zero hours with decimal"),
        ("2:30:15", 9015.0, "Multiple hours"),
        ("0:00:30", 30.0, "Just seconds in HH:MM:SS"),
        
        # Text format
        ("45s", 45.0, "Seconds only (text)"),
        ("2m15s", 135.0, "Minutes and seconds (text)"),
        ("1h23m45s", 5025.0, "Hours, minutes, seconds (text)"),
        ("1h30m", 5400.0, "Hours and minutes only (text)"),
        ("2h", 7200.0, "Hours only (text)"),
        ("90m", 5400.0, "Minutes only (text)"),
        ("1h23m45.5s", 5025.5, "Text format with decimal"),
        
        # Edge cases
        ("0", 0.0, "Zero"),
        ("0:00", 0.0, "Zero in MM:SS"),
        ("0:00:00", 0.0, "Zero in HH:MM:SS"),
    ]
    
    print("✅ VALID FORMATS:")
    print("-" * 50)
    
    success_count = 0
    for input_str, expected, description in test_cases:
        try:
            result = parse_smart_timestamp(input_str)
            if abs(result - expected) < 0.001:  # Allow for floating point precision
                formatted = format_seconds(result)
                print(f"  ✅ '{input_str}' → {result:.1f}s ({formatted}) - {description}")
                success_count += 1
            else:
                print(f"  ❌ '{input_str}' → {result:.1f}s (expected {expected:.1f}s) - {description}")
        except Exception as e:
            print(f"  ❌ '{input_str}' → ERROR: {e} - {description}")
    
    print(f"\n✅ Passed: {success_count}/{len(test_cases)} tests")
    
    # Test invalid formats
    print(f"\n❌ INVALID FORMATS (should fail):")
    print("-" * 50)
    
    invalid_cases = [
        "invalid",
        "1:2:3:4",  # Too many colons
        "abc:def",
        "1h2h3s",   # Duplicate units
        "",         # Empty string
        "1:60",     # Invalid seconds (60+)
        "25:00",    # Invalid minutes (25+ hours in MM:SS doesn't make sense)
    ]
    
    for invalid_input in invalid_cases:
        try:
            result = parse_smart_timestamp(invalid_input)
            print(f"  ⚠️  '{invalid_input}' → {result:.1f}s (should have failed!)")
        except ValueError as e:
            print(f"  ✅ '{invalid_input}' → Correctly rejected")
        except Exception as e:
            print(f"  ❌ '{invalid_input}' → Unexpected error: {e}")


def interactive_test():
    """Interactive timestamp parser test."""
    print(f"\n🎮 INTERACTIVE TEST")
    print("=" * 50)
    print("Enter timestamps to test (or 'quit' to exit):")
    print("Examples: 45.5, 1:23.5, 1:02:30, 1h23m45s")
    print()
    
    while True:
        try:
            user_input = input("📝 Enter timestamp: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_input:
                continue
                
            result = parse_smart_timestamp(user_input)
            formatted = format_seconds(result)
            print(f"   ✅ '{user_input}' → {result:.1f} seconds ({formatted})")
            print()
            
        except ValueError as e:
            print(f"   ❌ Error: {e}")
            print()
        except KeyboardInterrupt:
            break
    
    print("👋 Thanks for testing!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_test()
    else:
        test_timestamp_formats()
        
        print(f"\n💡 Want to try your own timestamps?")
        print(f"Run: python test_smart_timestamps.py --interactive")