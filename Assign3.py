#!/usr/bin/env python
# coding: utf-8

# # 1. What is the concept of an abstract superclass?

# An abstract superclass is a class that defines methods that are meant to be implemented by derived classes, but that do not have any implementation themselves.
# 
# In Python, you can define an abstract superclass using the abc (abstract base class) module. To do this, you define a class that derives from abc.ABC, and you decorate the methods that you want to be abstract with the @abc.abstractmethod decorator.
# The concept of an abstract superclass is useful when you want to define a common interface for a group of related classes, but you do not want to provide a concrete implementation for all of the methods. It allows you to specify the methods that derived classes must implement, while still allowing the derived classes to define their own specific behavior.

# # 2. What happens when a class statement's top level contains a basic assignment statement?

# If a class statement's top level (that is, the indented code block following the class definition) contains a basic assignment statement (that is, an assignment statement that is not part of a method definition), it will create a class attribute.

# # 3. Why does a class need to manually call a superclass's __init__ method?

# A class needs to manually call a superclass's __init__ method (that is, the method that is used to initialize the object) because the __init__ method is not automatically inherited by the subclass.
# 
# When you define a subclass, the subclass inherits the attributes and methods of the superclass, but it does not automatically inherit the __init__ method. This means that if you want to initialize the attributes of the subclass, you need to define an __init__ method in the subclass and call the superclass's __init__ method manually.

# # 4. How can you augment, instead of completely replacing, an inherited method?

# To augment an inherited method (that is, to add additional behavior to the method without completely replacing it), you can define a method in the subclass with the same name as the inherited method. When you call the method from the subclass, it will execute the code in the subclass's method in addition to the code in the inherited method.

# # 5. How is the local scope of a class different from that of a function?

# The local scope of a class is different from that of a function in several ways:
# 
# The local scope of a class includes all of the attributes and methods defined within the class definition, as well as any attributes or methods defined within the class's methods. The local scope of a function, on the other hand, only includes the variables defined within the function itself.
# 
# The local scope of a class is created when the class is defined, and it persists for the lifetime of the program. The local scope of a function, on the other hand, is created each time the function is called, and it is destroyed when the function returns.
# 
# The local scope of a class can be accessed from anywhere within the program, while the local scope of a function can only be accessed from within the function itself or from code that is nested within the function.
# 
# The local scope of a class is not affected by the global or nonlocal statements, which are used to access global and enclosing function variables from within a function. The local scope of a function, on the other hand, can be modified using these statements.
# 
# Overall, the local scope of a class is more persistent and more widely accessible than the local scope of a function. It is used to store attributes and methods that are associated with the class itself, rather than with any particular instance of the class.
# 
# 
# 
# 
# 
