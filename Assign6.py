#!/usr/bin/env python
# coding: utf-8

# # Q1. Describe three applications for exception processing.

# There are many applications for exception processing in programming, and some examples of common uses include:
# 
# Error handling: Exception processing can be used to handle errors that occur during the execution of a program. For example, if a program tries to read a file that does not exist, it might raise an exception to indicate that the file could not be found. The exception can then be caught and handled by the program, allowing it to recover from the error and continue executing.
# 
# Input validation: Exception processing can be used to validate user input and ensure that it meets certain criteria. For example, a program might raise an exception if the user provides an invalid email address or password. This can help to prevent errors and ensure that the program is processing valid data.
# 
# Resource management: Exception processing can be used to manage resources that are acquired during the execution of a program. For example, a program might open a database connection or allocate memory, and then raise an exception if it is unable to do so. The exception can be caught and used to release the resources and clean up before the program exits.
# 
# These are just a few examples of the many possible applications for exception processing in programming. Exception handling can be a powerful tool for managing errors, validating data, and managing resources in a program.
# 
# 
# 
# 
# 

# # Q2. What happens if you don't do something extra to treat an exception?

# If you don't do anything to handle an exception that is raised in a program, the program will terminate and produce an error message. This is because exceptions are used to indicate that something unexpected has occurred and that the program cannot continue executing normally. If an exception is not handled, it will propagate up the call stack until it is caught by the system or until the program terminates.
# 
# To prevent this from happening, it is usually necessary to do something extra to treat an exception, such as catching and handling it. This can be done using a try-except block, which allows you to specify a block of code that might raise an exception, and a separate block of code to execute if an exception is raised.
# 
# 
# By catching and handling the exception, you can prevent the program from terminating and allow it to recover from the error and continue executing.
# 
# 
# 
# 

# # Q3. What are your options for recovering from an exception in your script?

# There are several options for recovering from an exception in a script:
# 
# Catch and handle the exception: One option is to catch the exception using a try-except block, and then handle it by executing some code to recover from the error. For example, you might display an error message to the user or retry the operation that failed.
# 
# Ignore the exception: In some cases, you might decide to ignore the exception and continue executing the script as if nothing had happened. This is generally not recommended, as it can lead to unpredictable behavior and potentially hide important errors. However, it might be appropriate in some situations where the exception is not critical and does not affect the overall operation of the script.
# 
# Terminate the script: If you are unable to recover from the exception and the script is no longer able to function correctly, you might choose to terminate the script. This can be done by letting the exception propagate up the call stack and not catching it, or by explicitly exiting the script using a function like sys.exit().
# 
# Restart the script: In some cases, you might decide to restart the entire script if an exception is raised. This can be done by calling the script again using a function like os.execv() or subprocess.run().
# 
# Which option you choose will depend on the specific needs of your script and the type of exception that was raised. It is generally a good idea to try to handle exceptions as gracefully as possible, rather than simply ignoring or terminating the script.
# 
# 
# 
# 

# # Q4. Describe two methods for triggering exceptions in your script

# The first method for triggering exceptions in a script is to use the "raise" keyword. This keyword allows you to raise a specific exception, such as a ValueError or TypeError, and include a custom error message.
# The second method for triggering exceptions in a script is to use the "assert" keyword. This keyword allows you to make a statement that should evaluate to true, and if it evaluates to false, it raises an AssertionError with an optional error message.
# Both raise and assert statement can be use as a way to trigger exceptions, assert statement is more for testing/debugging and raise for specific error handling in specific case.
# x = 5
# if x > 10:
#     raise ValueError("x should not be greater than 10")
# x = 5
# assert x < 10, "x should be less than 10"
# 

# # Q5. Identify two methods for specifying actions to be executed at termination time, regardless of whether or not an exception exists.

# One method for specifying actions to be executed at termination time, regardless of whether or not an exception exists, is to use a try-finally block. The code within the "finally" block will always be executed, even if an exception is raised and caught within the "try" block.

# In[1]:


try:
    x = 5/0
except ZeroDivisionError:
    print("Caught division by zero")
finally:
    print("This will always be executed")


# Another method for specifying actions to be executed at termination time is to use a context manager with the 'with' statement. The 'with' statement is used to wrap the execution of a block of code, and an object which defined the special methods enter and exit is used. The __exit__ method will always be executed at the end of the block, regardless of whether an exception was raised or not.

# In[2]:


class MyContext:
    def __enter__(self):
        print("Entering the context")
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting the context")

with MyContext():
    x = 5/0


# In[ ]:




