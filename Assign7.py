#!/usr/bin/env python
# coding: utf-8

# # Q1. What is the purpose of the try statement?

# The purpose of the try statement is to define a block of code (referred to as the "try block") that may raise an exception. If an exception is raised within the try block, the code immediately following the try block (in the "except" block) will be executed in order to handle the exception. This allows for the program to continue running and avoid crashing, instead of halting execution as soon as an exception is encountered. The try statement also allows you to test a block of code for errors, and handle the errors in a separate block of code. This improves the readability and maintainability of the code.

# # Q2. What are the two most popular try statement variations?

# The first popular variation of the try statement is the try-except block. This variation allows you to catch specific exceptions that may be raised within the try block and handle them in a separate block of code

# In[1]:


try:
    x = 5/0
except ZeroDivisionError:
    print("Caught division by zero")


# The second popular variation of the try statement is the try-finally block. This variation allows you to specify a block of code (in the "finally" block) that will always be executed at the end of the try block, regardless of whether an exception is raised or not.
# 

# In[2]:


try:
    x = 5/0
except ZeroDivisionError:
    print("Caught division by zero")
finally:
    print("This will always be executed")


# Both try-except and try-finally are widely used variations of the try statement, they are powerful tools that help to handle exceptions and provide a way to specify actions to be executed at termination time.

# # Q3. What is the purpose of the raise statement?

# The purpose of the raise statement is to raise an exception. When an exception is raised, the normal flow of execution of the program is interrupted and the program jumps to the nearest exception handler that is capable of handling the exception. The raise statement allows you to trigger an exception explicitly in your code, rather than waiting for one to occur naturally. This can be useful for signaling error conditions or for enforcing preconditions in your code.
# 
# The raise statement takes one argument, which should be an exception type (for example, ValueError or TypeError) or an instance of a class derived from the BaseException class. You can also provide an optional second argument, which is a string that will be used as the error message for the exception.

# In[4]:


x = 11
if x > 10:
    raise ValueError("x should not be greater than 10")


# # Q4. What does the assert statement do, and what other statement is it like?

# The assert statement is used to make a statement that should evaluate to true, and if it evaluates to false, it raises an AssertionError with an optional error message. The assert statement is a debugging aid that tests a condition, and triggers an error if the condition is not true.

# In[5]:


x = 5
assert x < 10, "x should be less than 10"


# In[ ]:


Q5. What is the purpose of the with/as argument, and what other statement is it like


# In[ ]:


The 'with' statement is used to wrap the execution of a block of code and an object which defined the special methods __enter__ and __exit__ is used. The __enter__ method is run when the execution enters the block, and the __exit__ method is run when the execution leaves the block, regardless of whether an exception was raised or not.

The __exit__ method can also take 3 arguments (exc_type, exc_val, exc_tb) that allow you to access information about the exception and decide whether to suppress it or not.

The as argument is used to bind the result of the __enter__ method to a variable.


# In[ ]:


class MyContext:
    def __enter__(self):
        print("Entering the context")
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting the context")

with MyContext() as my_context:
    x = 5/0


# The 'with' statement is similar to the try-finally block in that both provide a way to specify actions to be executed at termination time. However, the 'with' statement is more powerful as it can specify the behavior of the context in case of an exception and bind the result of the __enter__ method to a variable. Additionally, the 'with' statement makes the code more readable and less prone to errors.
