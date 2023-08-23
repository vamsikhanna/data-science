#!/usr/bin/env python
# coding: utf-8

# # 1. Write a Python program to find sum of elements in list?

# In[1]:


# Create a list of numbers
numbers = [1, 2, 3, 4, 5]

# Use the sum() function to calculate the sum
sum_of_elements = sum(numbers)

# Print the sum
print("Sum of elements:", sum_of_elements)


# # 2. Write a Python program to Multiply all numbers in the list?
# Import the functools module to use reduce
from functools import reduce

# Create a list of numbers
numbers = [1, 2, 3, 4, 5]

# Initialize a variable to store the product, starting with 1
product_of_elements = 1

# Iterate through the list and multiply each element with the product
for num in numbers:
    product_of_elements *= num

# Print the product
print("Product of elements:", product_of_elements)

# # 3. Write a Python program to find smallest number in a list?

# In[ ]:


# Create a list of numbers
numbers = [10, 2, 45, 78, 31, 99, 4]

# Use the min() function to find the smallest number
smallest_number = min(numbers)

# Print the smallest number
print("Smallest number in the list:", smallest_number)


# # 4. Write a Python program to find largest number in a list?

# In[3]:


# Create a list of numbers
numbers = [10, 2, 45, 78, 31, 99, 4]

# Use the max() function to find the largest number
largest_number = max(numbers)

# Print the largest number
print("Largest number in the list:", largest_number)


# # 5. Write a Python program to find second largest number in a list?

# In[4]:


# Create a list of numbers
numbers = [10, 2, 45, 78, 31, 99, 4]

# Sort the list in descending order
sorted_numbers = sorted(numbers, reverse=True)

# Check if there are at least two elements in the list
if len(sorted_numbers) >= 2:
    # The second largest number is at index 1 in the sorted list
    second_largest = sorted_numbers[1]
    print("Second largest number in the list:", second_largest)
else:
    print("The list does not have at least two elements.")


# # 6. Write a Python program to find N largest elements from a list?

# In[5]:


import heapq

# Create a list of numbers
numbers = [10, 2, 45, 78, 31, 99, 4]

# Define N (the number of largest elements to find)
N = 3

# Use the heapq.nlargest() function to find the N largest elements
largest_elements = heapq.nlargest(N, numbers)

# Print the N largest elements
print(f"{N} largest elements in the list:", largest_elements)


# # 7. Write a Python program to print even numbers in a list?

# In[10]:


# Create a list of numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Use list comprehension to filter even numbers
even_numbers = [num for num in numbers if num % 2 == 0]

# Print the even numbers
print("Even numbers in the list:", even_numbers)


# # 8. Write a Python program to print odd numbers in a List?

# In[9]:


# Create a list of numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Use list comprehension to filter odd numbers
odd_numbers = [num for num in numbers if num % 2 != 0]

# Print the odd numbers
print("Odd numbers in the list:", odd_numbers)


# # 9. Write a Python program to Remove empty List from List?

# In[8]:


# Create a list with empty and non-empty lists
my_list = [1, [], 2, [], [], 3, 4, [], 5]

# Use list comprehension to remove empty lists
filtered_list = [sublist for sublist in my_list if sublist]

# Print the filtered list
print("Original List:", my_list)
print("List with Empty Lists Removed:", filtered_list)


# # 10. Write a Python program to Cloning or Copying a list?

# In[7]:


# Create a list to be cloned
original_list = [1, 2, 3, 4, 5]

# Use the list() constructor to create a new copy of the list
cloned_list = list(original_list)

# Modify the cloned_list (optional)
cloned_list.append(6)

# Print both the original and cloned lists
print("Original List:", original_list)
print("Cloned List:", cloned_list)


# # 11. Write a Python program to Count occurrences of an element in a list?

# In[6]:


# Create a list of elements
my_list = [1, 2, 2, 3, 4, 2, 5, 2]

# Define the element you want to count
element_to_count = 2

# Use the count() method to count occurrences of the element
count = my_list.count(element_to_count)

# Print the count
print(f"The element {element_to_count} appears {count} times in the list.")


# In[ ]:




