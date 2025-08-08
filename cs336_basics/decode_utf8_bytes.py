def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    """
    INCORRECT implementation of UTF-8 decoding.
    
    This function is wrong because it treats each byte individually,
    which doesn't work for UTF-8 encoding where characters can span multiple bytes.
    
    Args:
        bytestring: Input bytes to decode
        
    Returns:
        String (but may be incorrect for multi-byte UTF-8 characters)
    """
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])


def decode_utf8_bytes_to_str_correct(bytestring: bytes):
    """
    CORRECT implementation of UTF-8 decoding.
    
    This function properly decodes UTF-8 bytes by treating them as a complete
    UTF-8 sequence rather than individual bytes.
    
    Args:
        bytestring: Input bytes to decode
        
    Returns:
        Properly decoded UTF-8 string
        
    Raises:
        UnicodeDecodeError: If the bytes are not valid UTF-8
    """
    return bytestring.decode("utf-8")


def decode_utf8_bytes_to_str_with_errors(bytestring: bytes, errors: str = "strict"):
    """
    UTF-8 decoding with configurable error handling.
    
    Args:
        bytestring: Input bytes to decode
        errors: Error handling strategy ('strict', 'ignore', 'replace', 'backslashreplace')
        
    Returns:
        Decoded string according to the error handling strategy
    """
    return bytestring.decode("utf-8", errors=errors)


def demonstrate_utf8_decoding():
    """
    Demonstrate the difference between correct and incorrect UTF-8 decoding.
    """
    # Test cases with different types of UTF-8 sequences
    test_cases = [
        b"Hello, World!",  # ASCII only
        b"Hello, \xe4\xb8\x96\xe7\x95\x8c!",  # ASCII + UTF-8 (世界 = "world" in Chinese)
        b"\xf0\x9f\x98\x80",  # Emoji (😀)
        b"Mixed: \xe4\xb8\x96\xe7\x95\x8c \xf0\x9f\x98\x80",  # Mixed content
    ]
    
    print("UTF-8 Decoding Demonstration")
    print("=" * 50)
    
    for i, test_bytes in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_bytes}")
        print(f"Input bytes: {test_bytes}")
        
        try:
            wrong_result = decode_utf8_bytes_to_str_wrong(test_bytes)
            print(f"Wrong method result: {wrong_result}")
        except UnicodeDecodeError as e:
            print(f"Wrong method error: {e}")
        
        try:
            correct_result = decode_utf8_bytes_to_str_correct(test_bytes)
            print(f"Correct method result: {correct_result}")
        except UnicodeDecodeError as e:
            print(f"Correct method error: {e}")
        
        # Show with error handling
        ignore_result = decode_utf8_bytes_to_str_with_errors(test_bytes, errors="ignore")
        print(f"With ignore errors: {ignore_result}")
        
        replace_result = decode_utf8_bytes_to_str_with_errors(test_bytes, errors="replace")
        print(f"With replace errors: {replace_result}")


def test_invalid_utf8():
    """
    Test handling of invalid UTF-8 sequences.
    """
    print("\n" + "=" * 50)
    print("Testing Invalid UTF-8 Sequences")
    print("=" * 50)
    
    # Invalid UTF-8 sequences
    invalid_cases = [
        b"\xff\xfe",  # Invalid UTF-8 bytes
        b"Hello\xffWorld",  # Mixed valid and invalid
        b"\xc0\xaf",  # Overlong encoding
    ]
    
    for i, invalid_bytes in enumerate(invalid_cases, 1):
        print(f"\nInvalid Test {i}: {invalid_bytes}")
        
        try:
            result = decode_utf8_bytes_to_str_correct(invalid_bytes)
            print(f"Strict decoding: {result}")
        except UnicodeDecodeError as e:
            print(f"Strict decoding error: {e}")
        
        ignore_result = decode_utf8_bytes_to_str_with_errors(invalid_bytes, errors="ignore")
        print(f"Ignore errors: {ignore_result}")
        
        replace_result = decode_utf8_bytes_to_str_with_errors(invalid_bytes, errors="replace")
        print(f"Replace errors: {replace_result}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_utf8_decoding()
    test_invalid_utf8()