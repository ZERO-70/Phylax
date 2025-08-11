#!/usr/bin/env python3

import re

def clean_existing_transcript():
    """Clean profanity from existing transcript file"""
    
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
    
    # Read existing transcript
    try:
        with open('audio_transcript.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Filter profanity from content
        filtered_content = filter_profanity(content)
        
        # Write back the filtered content
        with open('audio_transcript.txt', 'w', encoding='utf-8') as f:
            f.write(filtered_content)
        
        print("✓ Existing transcript has been cleaned of profanity")
        
        # Show what was changed
        if content != filtered_content:
            print("Changes made:")
            lines_original = content.split('\n')
            lines_filtered = filtered_content.split('\n')
            
            for i, (orig, filt) in enumerate(zip(lines_original, lines_filtered)):
                if orig != filt:
                    print(f"Line {i+1}: '{orig.strip()}' -> '{filt.strip()}'")
        else:
            print("No profanity found in existing transcript")
            
    except FileNotFoundError:
        print("No existing transcript file found")
    except Exception as e:
        print(f"Error cleaning transcript: {e}")

if __name__ == "__main__":
    clean_existing_transcript()
