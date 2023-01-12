#!/usr/bin/env python
# coding: utf-8

# # Q1. What are the two latest user-defined exception constraints in Python 3.X?
# 
# 

# In Python 3.X, two of the latest user-defined exception constraints are:
# 
# BaseException: All user-defined exceptions should inherit from BaseException or one of its subclasses. This is to ensure that user-defined exceptions are consistent with built-in exceptions and are caught by the correct catch block.
# 
# Exception hierarchy: User-defined exceptions should be organized in a hierarchy and should inherit from a more general exception class. This allows developers to catch more specific exceptions while also catching more general ones. It makes the code more readable, maintainable and less prone to errors.

# # Q2. How are class-based exceptions that have been raised matched to handlers?
# 
# 

# In Python, class-based exceptions that have been raised are matched to handlers based on the class hierarchy of the exception. When an exception is raised, the interpreter starts looking for an exception handler that can handle the exception by looking at the class hierarchy of the exception. The interpreter begins by looking for a handler for the exact class of the exception. If no handler is found, the interpreter looks for a handler for the immediate superclass of the exception. This process continues until the interpreter finds a matching handler or reaches the BaseException class, which is the base class for all exceptions.
# 
# It's worth noting that the interpreter looks for handlers in the order they appear in the code. The first handler that matches the raised exception will handle it. When a handler is found, the interpreter stops looking for more handlers and the code in the handler block will be executed.
# 
# For example, if you raise a ValueError exception, the interpreter will first look for a handler for the ValueError class, and if it doesn't find one, it will look for a handler for the Exception class (which is the superclass of ValueError) and finally for the BaseException class. If a matching handler is not found, the interpreter will stop the execution of the program and print the traceback of the exception.
# 
# Additionally, it's worth noting that the interpreter will not look for handlers for the BaseException class, it will stop the execution of the program and print the traceback of the exception.
# 
# 
# 
# 
# 

# # Q3. Describe two methods for attaching context information to exception artefacts.

# One method for attaching context information to exception artifacts is to use the with_traceback() method. This method allows you to attach a traceback to an exception, which is useful for providing additional context about the exception. This method takes one argument, which should be a traceback object, and returns a new exception object with the traceback attached.

# In[1]:


try:
    x = 5/0
except ZeroDivisionError as e:
    e = e.with_traceback(sys.exc_info()[2])
    raise e


# Another method for attaching context information to exception artifacts is to use the raise Exception("message") from original_exception statement. This statement allows you to raise a new exception while preserving the original exception traceback, and also providing an additional error message. This statement allows you to attach the context information to the original exception without losing the traceback

# In[2]:


try:
    x = 5/0
except ZeroDivisionError as e:
    raise Exception("Error occurred while performing division") from e


# # Q4. Describe two methods for specifying the text of an exception object's error message.
# 
# 

# One method for specifying the text of an exception object's error message is to include it as an argument when raising the exception.

# In[ ]:


x = 5
if x > 10:
    raise ValueError("x should not be greater than 10")


# Another method for specifying the text of an exception object's error message is to assign it to the args attribute of the exception object. The args attribute is a tuple, and the first element is usually used as the error message

# In[3]:


try:
    x = 5/0
except ZeroDivisionError as e:
    e.args = ("Error occurred while performing division",)
    raise e


# # Q5. Why do you no longer use string-based exceptions?

# String-based exceptions are no longer used in Python because they have several drawbacks compared to class-based exceptions.
# 
# String-based exceptions are not objects, therefore they don't have the ability to inherit from other exceptions or to be subclassed. This makes it harder to create a hierarchy of exceptions and to catch specific exceptions.
# 
# String-based exceptions can be easily mistyped and are not enforced by the interpreter. This can lead to hard-to-find bugs in the code, as the interpreter will not raise an error if a non-existent string exception is raised.
# 
# String-based exceptions do not provide any additional information about the exception other than the error message, whereas class-based exceptions can carry additional information, such as stack trace, attributes, etc.
# 
# String-based exceptions are not easily extensible and cannot be subclassed, which means that it's harder to add new functionality to them.
# 
# For these reasons, class-based exceptions are preferred in Python. They are objects, have the ability to inherit from other exceptions, are enforced by the interpreter, provide additional information, and can be easily extended with custom functionality

# In[ ]:




