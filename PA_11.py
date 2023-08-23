#!/usr/bin/env python
# coding: utf-8

# # 1. Write a Python program to find words which are greater than given length k?

# In[2]:


# Create a list of words
word_list = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]

# Define the length threshold 'k'
k = 5

# Use list comprehension to find words longer than 'k'
long_words = [word for word in word_list if len(word) > k]

# Print the long words
print(f"Words longer than {k} characters: {long_words}")


# # 2. Write a Python program for removing i-th character from a string?

# In[3]:


# Input string
input_string = "Hello, World!"

# Define the index (i) of the character to remove
i = 3  # This will remove the 4th character (0-based index)

# Check if the index is valid
if 0 <= i < len(input_string):
    # Remove the i-th character by slicing the string
    modified_string = input_string[:i] + input_string[i + 1:]
    print("Modified string:", modified_string)
else:
    print("Invalid index. The string remains unchanged.")


# # 3. Write a Python program to split and join a string?

# In[4]:


# Input string
input_string = "This is a sample sentence."

# Split the string into words using space as the delimiter
words = input_string.split()

# Join the words into a new string using a hyphen as the separator
joined_string = "-".join(words)

# Print the split and joined strings
print("Original string:", input_string)
print("Split into words:", words)
print("Joined with hyphen:", joined_string)


# # 4. Write a Python to check if a given string is binary string or not?

# In[5]:


# Function to check if a string is a binary string
def is_binary_string(input_string):
    # Loop through each character in the string
    for char in input_string:
        # If the character is not '0' or '1', it's not a binary string
        if char not in ('0', '1'):
            return False
    # If all characters are '0' or '1', it's a binary string
    return True

# Input string to be checked
input_string = "1010101010"

# Check if the input string is a binary string
if is_binary_string(input_string):
    print(f"'{input_string}' is a binary string.")
else:
    print(f"'{input_string}' is not a binary string.")


# # 5. Write a Python program to find uncommon words from two Strings?

# In[6]:


# Function to find uncommon words from two strings
def find_uncommon_words(string1, string2):
    # Split the strings into words using space as the delimiter
    words1 = string1.split()
    words2 = string2.split()

    # Convert the word lists to sets for efficient comparison
    set1 = set(words1)
    set2 = set(words2)

    # Find uncommon words by taking the symmetric difference of the sets
    uncommon_words = set1.symmetric_difference(set2)

    return list(uncommon_words)

# Input strings
string1 = "This is the first string with some words."
string2 = "This is the second string with different words."

# Find uncommon words
uncommon_words = find_uncommon_words(string1, string2)

# Print the uncommon words
print("Uncommon words:", uncommon_words)


# # 6. Write a Python to find all duplicate characters in string?

# In[7]:


# Function to find duplicate characters in a string
def find_duplicate_characters(input_string):
    # Create an empty dictionary to store character counts
    char_count = {}

    # Initialize a list to store duplicate characters
    duplicates = []

    # Iterate through the string
    for char in input_string:
        # Increment the count for the character in the dictionary
        char_count[char] = char_count.get(char, 0) + 1

    # Check for characters with counts greater than 1 (duplicates)
    for char, count in char_count.items():
        if count > 1:
            duplicates.append(char)

    return duplicates

# Input string
input_string = "programming"

# Find duplicate characters
duplicate_chars = find_duplicate_characters(input_string)

# Print the duplicate characters
print("Duplicate characters:", duplicate_chars)


# # 7. Write a Python Program to check if a string contains any special character?

# In[8]:


import re

# Function to check if a string contains any special characters
def contains_special_characters(input_string):
    # Define a regular expression pattern to match special characters
    pattern = r'[!@#$%^&*()_+{}\[\]:;<>,.?~\\|/]'
    
    # Use re.search() to find a match
    match = re.search(pattern, input_string)
    
    # If a match is found, it contains special characters
    if match:
        return True
    else:
        return False

# Input string
input_string = "Hello, World!"

# Check if the input string contains special characters
if contains_special_characters(input_string):
    print(f"'{input_string}' contains special characters.")
else:
    print(f"'{input_string}' does not contain special characters.")


# In[ ]:




