#!/usr/bin/env python3

import re
import time

def test_abusive_detection():
    """Test the new abusive language detection system"""
    
    # Load profanity words from external dataset
    profanity_words = []
    try:
        with open('profanity_wordlist.txt', 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word and len(word) > 1:
                    profanity_words.append(word)
        print(f"✓ Loaded {len(profanity_words)} profanity words from dataset")
    except FileNotFoundError:
        print("✗ Profanity wordlist not found")
        return
    
    # Create regex pattern
    pattern = r'\b(' + '|'.join(re.escape(word) for word in profanity_words) + r')\b'
    profanity_pattern = re.compile(pattern, re.IGNORECASE)
    
    # Simulate temperature tracking
    abusive_temperature = 0.0
    decay_factor = 1.3
    last_check_time = time.time()
    
    def detect_abusive_words_realtime(text):
        nonlocal abusive_temperature, last_check_time
        
        # Cool down temperature before each check
        current_time = time.time()
        time_elapsed = current_time - last_check_time
        if time_elapsed > 0:
            abusive_temperature /= (decay_factor ** time_elapsed)
            if abusive_temperature < 0.1:
                abusive_temperature = 0.0
        
        last_check_time = current_time
        
        # Find abusive words
        matches = profanity_pattern.findall(text.lower())
        abusive_count = len(matches)
        
        if abusive_count > 0:
            abusive_temperature += abusive_count
            abusive_temperature = min(100.0, abusive_temperature)
            print(f"  🔥 Detected {abusive_count} abusive words: {list(set(matches))}")
            print(f"  📊 Temperature: {abusive_temperature:.1f}°")
        else:
            print(f"  ✓ Clean text, temperature: {abusive_temperature:.1f}°")
        
        return abusive_count
    
    # Test cases
    test_cases = [
        "This is a normal sentence",
        "Like shut the fuck up",
        "What the hell is going on",
        "This is another clean sentence",
        "That's complete bullshit, you idiot",
        "Let me wait a moment...",  # For cooling down
        "Just a regular conversation"
    ]
    
    print("\n" + "="*50)
    print("🧪 TESTING ABUSIVE LANGUAGE DETECTION SYSTEM")
    print("="*50)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n[Test {i}] '{test_text}'")
        
        if "wait a moment" in test_text:
            print("  ⏱️  Simulating 1-second pause for cooling...")
            time.sleep(1)
        
        count = detect_abusive_words_realtime(test_text)
    
    print(f"\n🏁 Final temperature: {abusive_temperature:.1f}°")
    print("✅ Abusive language detection test completed!")

if __name__ == "__main__":
    test_abusive_detection()
