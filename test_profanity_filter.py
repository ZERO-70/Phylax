#!/usr/bin/env python3

import re

def test_profanity_filter():
    """Test the profanity filter with sample text"""
    
    # Initialize profanity filter (same as in receiver.py)
    profanity_words = [
        # Strong profanity
        'fuck', 'fucking', 'fucked', 'fucker', 'fucks',
        'shit', 'shitting', 'shits', 'bullshit',
        'bitch', 'bitches', 'bitching',
        'damn', 'damned', 'dammit',
        'ass', 'asshole', 'asses',
        'bastard', 'bastards',
        'crap', 'crappy',
        'piss', 'pissed', 'pissing',
        'hell',
        # Slurs and offensive terms
        'retard', 'retarded',
        'idiot', 'idiotic', 'moron', 'stupid',
    ]
    
    # Create regex pattern for case-insensitive matching
    pattern = r'\b(' + '|'.join(re.escape(word) for word in profanity_words) + r')\b'
    profanity_pattern = re.compile(pattern, re.IGNORECASE)
    
    def filter_profanity(text):
        """Filter profanity from text, replacing with **word** format"""
        if not text:
            return text
        
        def replace_profanity(match):
            word = match.group(0)
            # Keep first and last character, replace middle with asterisks
            if len(word) == 1:
                return f"**{word}**"
            elif len(word) == 2:
                return f"**{word[0]}{word[1]}**"
            else:
                return f"**{word[0]}{'*' * (len(word) - 2)}{word[-1]}**"
        
        filtered_text = profanity_pattern.sub(replace_profanity, text)
        return filtered_text
    
    # Test with existing transcript content
    test_text = "Like shut the fuck up"
    print(f"Original: '{test_text}'")
    print(f"Filtered: '{filter_profanity(test_text)}'")
    print()
    
    # Test with various profanity
    test_cases = [
        "This is fucking crazy",
        "What the hell is going on",
        "That's bullshit",
        "Don't be an idiot",
        "This is a normal sentence",
        "Damn this is bad",
        "You're such an asshole",
    ]
    
    print("Testing profanity filter:")
    print("-" * 50)
    
    for test in test_cases:
        filtered = filter_profanity(test)
        status = "FILTERED" if test != filtered else "CLEAN"
        print(f"[{status}] '{test}' -> '{filtered}'")

if __name__ == "__main__":
    test_profanity_filter()
