#!/usr/bin/env python3
"""
Test Script for Configuration System

This script tests the configuration system to ensure all features
can be properly enabled/disabled and configured.
"""

import sys
import os
import tempfile
import json
from config import Config, parse_command_line_args, apply_cli_overrides, DEFAULT_CONFIG

def test_config_creation():
    """Test basic configuration creation"""
    print("Testing configuration creation...")
    
    # Test default config
    config = Config()
    assert config.is_enabled('nsfw_detection'), "NSFW detection should be enabled by default"
    assert config.is_enabled('gun_detection'), "Gun detection should be enabled by default"
    assert config.is_enabled('transcription'), "Transcription should be enabled by default"
    assert config.is_enabled('profanity_filter'), "Profanity filter should be enabled by default"
    
    print("✓ Default configuration created successfully")

def test_config_file_operations():
    """Test saving and loading configuration files"""
    print("Testing configuration file operations...")
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_config_file = f.name
    
    try:
        # Create config and modify it
        config = Config(temp_config_file)
        config.set('nsfw_detection', 'enabled', False)
        config.set('transcription', 'whisper_model', 'tiny')
        config.save_config()
        
        # Load it again and verify changes
        config2 = Config(temp_config_file)
        assert not config2.is_enabled('nsfw_detection'), "NSFW should be disabled"
        assert config2.get('transcription', 'whisper_model') == 'tiny', "Whisper model should be tiny"
        
        print("✓ Configuration file save/load works correctly")
        
    finally:
        # Clean up
        if os.path.exists(temp_config_file):
            os.unlink(temp_config_file)

def test_command_line_parsing():
    """Test command line argument parsing"""
    print("Testing command line argument parsing...")
    
    # Mock sys.argv for testing
    original_argv = sys.argv
    
    try:
        # Test disable flags
        sys.argv = ['test', '--no-nsfw', '--no-gun']
        args = parse_command_line_args()
        assert args.no_nsfw, "no_nsfw flag should be set"
        assert args.no_gun, "no_gun flag should be set"
        
        # Test enable flags
        sys.argv = ['test', '--enable-transcription', '--whisper-model', 'base']
        args = parse_command_line_args()
        assert args.enable_transcription, "enable_transcription flag should be set"
        assert args.whisper_model == 'base', "whisper_model should be base"
        
        print("✓ Command line parsing works correctly")
        
    finally:
        sys.argv = original_argv

def test_cli_overrides():
    """Test command line overrides on configuration"""
    print("Testing command line overrides...")
    
    # Create config with defaults
    config = Config()
    
    # Mock command line arguments
    class MockArgs:
        def __init__(self):
            self.no_nsfw = True
            self.no_gun = False
            self.no_transcription = False
            self.no_profanity = True
            self.enable_nsfw = False
            self.enable_gun = False
            self.enable_transcription = True
            self.enable_profanity = False
            self.host = '192.168.1.100'
            self.video_port = 8080
            self.audio_port = 8081
            self.whisper_model = 'tiny'
    
    args = MockArgs()
    apply_cli_overrides(config, args)
    
    # Verify overrides
    assert not config.is_enabled('nsfw_detection'), "NSFW should be disabled by --no-nsfw"
    assert not config.is_enabled('profanity_filter'), "Profanity filter should be disabled"
    assert config.is_enabled('transcription'), "Transcription should be enabled by --enable-transcription"
    assert config.get('network', 'host') == '192.168.1.100', "Host should be overridden"
    assert config.get('network', 'video_port') == 8080, "Video port should be overridden"
    assert config.get('transcription', 'whisper_model') == 'tiny', "Whisper model should be overridden"
    
    print("✓ Command line overrides work correctly")

def test_configuration_merging():
    """Test configuration merging with defaults"""
    print("Testing configuration merging...")
    
    # Create partial config
    partial_config = {
        "nsfw_detection": {
            "enabled": False
        },
        "transcription": {
            "whisper_model": "large"
        }
    }
    
    # Create temporary file with partial config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(partial_config, f)
        temp_file = f.name
    
    try:
        # Load config and verify merging
        config = Config(temp_file)
        
        # Should have custom values
        assert not config.is_enabled('nsfw_detection'), "NSFW should be disabled from custom config"
        assert config.get('transcription', 'whisper_model') == 'large', "Whisper model should be large"
        
        # Should have default values for missing keys
        assert config.is_enabled('gun_detection'), "Gun detection should have default value"
        assert config.get('network', 'video_port') == 9999, "Video port should have default value"
        
        print("✓ Configuration merging works correctly")
        
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def test_feature_toggles():
    """Test all feature toggle combinations"""
    print("Testing feature toggle combinations...")
    
    config = Config()
    
    # Test all features disabled
    for feature in ['nsfw_detection', 'gun_detection', 'transcription', 'profanity_filter']:
        config.set(feature, 'enabled', False)
        assert not config.is_enabled(feature), f"{feature} should be disabled"
    
    # Test all features enabled
    for feature in ['nsfw_detection', 'gun_detection', 'transcription', 'profanity_filter']:
        config.set(feature, 'enabled', True)
        assert config.is_enabled(feature), f"{feature} should be enabled"
    
    print("✓ Feature toggles work correctly")

def test_invalid_configurations():
    """Test handling of invalid configurations"""
    print("Testing invalid configuration handling...")
    
    config = Config()
    
    # Test accessing non-existent sections
    result = config.get('nonexistent_section', 'key')
    assert result is None, "Non-existent section should return None"
    
    # Test accessing non-existent keys
    result = config.get('network', 'nonexistent_key')
    assert result is None, "Non-existent key should return None"
    
    # Test is_enabled on non-existent feature
    result = config.is_enabled('nonexistent_feature')
    assert not result, "Non-existent feature should return False"
    
    print("✓ Invalid configuration handling works correctly")

def run_all_tests():
    """Run all configuration tests"""
    print("Running Configuration System Tests")
    print("=" * 40)
    print()
    
    tests = [
        test_config_creation,
        test_config_file_operations,
        test_command_line_parsing,
        test_cli_overrides,
        test_configuration_merging,
        test_feature_toggles,
        test_invalid_configurations
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        print()
    
    print("Test Results")
    print("-" * 20)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {len(tests)}")
    
    if failed == 0:
        print("\\n🎉 All tests passed! Configuration system is working correctly.")
        return True
    else:
        print(f"\\n❌ {failed} test(s) failed. Please check the configuration system.")
        return False

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\\nTests interrupted by user")
        sys.exit(1)
