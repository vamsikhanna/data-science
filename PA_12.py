#!/usr/bin/env python
# coding: utf-8

# # 1. Write a Python program to Extract Unique values dictionary values?

# In[1]:


# Function to extract unique values from dictionary values
def extract_unique_values(dictionary):
    # Create an empty set to store unique values
    unique_values = set()

    # Iterate through the dictionary values
    for values in dictionary.values():
        # Add each value to the set
        unique_values.update(values)

    return list(unique_values)

# Input dictionary with values as lists
my_dict = {
    'A': [1, 2, 3],
    'B': [2, 3, 4],
    'C': [3, 4, 5]
}

# Extract unique values from dictionary values
unique_values = extract_unique_values(my_dict)

# Print the unique values
print("Unique values:", unique_values)


# # 2. Write a Python program to find the sum of all items in a dictionary?

# In[2]:


# Function to find the sum of all items in a dictionary
def sum_dictionary_values(dictionary):
    # Initialize a variable to store the sum
    total_sum = 0

    # Iterate through the dictionary values
    for value in dictionary.values():
        # Check if the value is a number (int or float)
        if isinstance(value, (int, float)):
            total_sum += value

    return total_sum

# Input dictionary with values of various data types
my_dict = {
    'A': 10,
    'B': 20.5,
    'C': "Hello",
    'D': 30,
    'E': 15.75
}

# Find the sum of all items in the dictionary
result = sum_dictionary_values(my_dict)

# Print the sum
print("Sum of dictionary values:", result)


# # 3. Write a Python program to Merging two Dictionaries?

# In[3]:


# Function to merge two dictionaries using the update() method
def merge_dicts_update(dict1, dict2):
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict

# Function to merge two dictionaries using the ** unpacking operator
def merge_dicts_unpacking(dict1, dict2):
    merged_dict = {**dict1, **dict2}
    return merged_dict

# Input dictionaries
dict1 = {'A': 1, 'B': 2}
dict2 = {'C': 3, 'D': 4}

# Merge dictionaries using the update() method
merged_dict1 = merge_dicts_update(dict1, dict2)
print("Merged dictionary using update():", merged_dict1)

# Merge dictionaries using the ** unpacking operator
merged_dict2 = merge_dicts_unpacking(dict1, dict2)
print("Merged dictionary using unpacking operator:", merged_dict2)


# # 4. Write a Python program to convert key-values list to flat dictionary?

# In[4]:


# Function to convert a list of key-value pairs to a flat dictionary
def list_to_flat_dict(key_value_list):
    flat_dict = {}
    
    for pair in key_value_list:
        if len(pair) == 2:
            key, value = pair
            flat_dict[key] = value
    
    return flat_dict

# Input list of key-value pairs
key_value_list = [["A", 1], ["B", 2], ["C", 3]]

# Convert the list to a flat dictionary
flat_dictionary = list_to_flat_dict(key_value_list)

# Print the flat dictionary
print("Flat Dictionary:", flat_dictionary)


# # 5. Write a Python program to insertion at the beginning in OrderedDict?

# In[5]:


from collections import OrderedDict

# Function to insert a key-value pair at the beginning of an OrderedDict
def insert_at_beginning(ordered_dict, key, value):
    # Move the existing key to the end (if it exists)
    if key in ordered_dict:
        ordered_dict.move_to_end(key, last=False)
    
    # Add the new key-value pair at the beginning
    ordered_dict[key] = value

# Create an example OrderedDict
ordered_dict = OrderedDict([('A', 1), ('B', 2), ('C', 3)])

# Insert a new key-value pair at the beginning
insert_at_beginning(ordered_dict, 'X', 10)

# Print the updated OrderedDict
print("Updated OrderedDict:", ordered_dict)


# # 6. Write a Python program to check order of character in string using OrderedDict()?

# In[6]:


from collections import OrderedDict

# Function to check if the characters in a string appear in order
def check_order_of_characters(input_string, test_string):
    # Create an OrderedDict to store the characters in the input_string
    char_dict = OrderedDict.fromkeys(input_string)

    # Initialize an index to track the position in the test_string
    index = 0

    # Iterate through the characters in the OrderedDict
    for char in char_dict:
        # Check if the current character matches the character in the test_string
        if char == test_string[index]:
            index += 1
        # If not, the order is not maintained
        else:
            return False

    # If all characters match in order, return True
    return True

# Input string and test string
input_string = "hello"
test_string = "hlo"

# Check if the characters in the test string appear in order in the input string
result = check_order_of_characters(input_string, test_string)

# Print the result
if result:
    print(f"The characters in '{test_string}' appear in order in '{input_string}'.")
else:
    print(f"The characters in '{test_string}' do not appear in order in '{input_string}'.")


# # 7.Write a Python program to sort Python Dictionaries by Key or Value?

# In[7]:


# Function to sort a dictionary by keys
def sort_dict_by_keys(input_dict):
    sorted_dict = dict(sorted(input_dict.items()))
    return sorted_dict

# Function to sort a dictionary by values
def sort_dict_by_values(input_dict):
    sorted_dict = dict(sorted(input_dict.items(), key=lambda item: item[1]))
    return sorted_dict

# Input dictionary
my_dict = {'B': 2, 'A': 1, 'D': 4, 'C': 3}

# Sort the dictionary by keys
sorted_by_keys = sort_dict_by_keys(my_dict)
print("Sorted by keys:", sorted_by_keys)

# Sort the dictionary by values
sorted_by_values = sort_dict_by_values(my_dict)
print("Sorted by values:", sorted_by_values)


# In[ ]:




