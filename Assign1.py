#!/usr/bin/env python
# coding: utf-8

# # Q1. What is the purpose of Python's OOP?
# 
# 

# Object-oriented programming (OOP) is a programming paradigm that is based on the concept of "objects", which can contain data and code that operates on that data. The main purpose of OOP in Python is to help programmers design and organize their code in a more logical, reusable, and maintainable way. OOP allows you to create "blueprints" for objects (called classes) and then create instances of those objects (called instances). This allows you to write code that is easier to understand and maintain, and it also allows you to reuse code across multiple projects. OOP is a powerful programming paradigm that is widely used in Python and many other programming languages

# # Q2. Where does an inheritance search look for an attribute?

# In Python, when you use inheritance, the search for an attribute starts in the subclass and then proceeds up the chain of base classes. This is known as the "method resolution order" (MRO).
# 
# 

# In[3]:


class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")

class C(A):
    def method(self):
        print("C")

class D(C, B):
    pass

d = D()
d.method()


# In[2]:


print(D.__mro__)


# # Q3. How do you distinguish between a class object and an instance object?
# 
# 

# In object-oriented programming, a class is a blueprint for creating objects. An instance is a specific object created from a class.

# In[4]:


class Dog:
  def __init__(self, name, breed):
    self.name = name
    self.breed = breed

dog1 = Dog("Fido", "Labrador")
dog2 = Dog("Buddy", "Poodle")


# # Q4. What makes the first argument in a class’s method function special?

# In a class method in Python, the first argument is usually named self. This argument is used to access the attributes and methods of the class instance

# In[5]:


class Dog:
  def __init__(self, name, breed):
    self.name = name
    self.breed = breed
    
  def bark(self):
    print("Woof!")

dog1 = Dog("Fido", "Labrador")
dog1.bark()  # Output: "Woof!"


# # Q5. What is the purpose of the __init__ method?
# 
# 

# The __init__ method is a special method in Python classes that is called when an instance of the class is created. It is used to initialize the attributes of the instance.
# The __init__ method is important because it allows you to create instances of the class with different attributes. Without the __init__ method, you would have to create a separate method to initialize the attributes for each instance you create.
# 
# The __init__ method is also sometimes called a constructor, because it is used to construct or create the instance of the class.

# In[6]:


class Dog:
  def __init__(self, name, breed):
    self.name = name
    self.breed = breed
    
dog1 = Dog("Fido", "Labrador")
print(dog1.name)  # Output: "Fido"
print(dog1.breed)  # Output: "Labrador"


# # Q6. What is the process for creating a class instance?

# To create an instance of a class in Python, you use the class name followed by parentheses. Optionally, you can pass in arguments to the class's constructor (the __init__ method) inside the parentheses

# # Q7. What is the process for creating a class?

# To create a class in Python, you use the class keyword, followed by the name of the class, and a colon. The class definition should include an __init__ method, which is a special method in Python classes that is called when an instance of the class is created.

# # Q8. How would you define the superclasses of a class?

# In Python, you can use inheritance to create a new class that is a modified version of an existing class. The existing class is called the superclass, and the new class is called the subclass.
# 
# To define a subclass that inherits from a superclass, you use the name of the superclass as a parameter in the definition of the subclass

# In[ ]:




