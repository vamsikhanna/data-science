#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('pinfo', '__getattribute__')


# `__getattr__` and `__getattribute__` are two special methods in Python that are related to attribute access, but they serve different purposes and are used in different ways:
# 
# 1. **`__getattr__` Method**:
# 
#    - Purpose: `__getattr__` is called when an attempt is made to access an attribute that doesn't exist in an object.
#    - Use Case: It's typically used to define custom behavior when an undefined attribute is accessed.
#    - Behavior: This method takes two arguments, `self` (the instance) and the name of the attribute being accessed. It can return a computed value, raise an `AttributeError`, or perform any other custom action.
#    - Example:
# 
#      ```python
#      class MyClass:
#          def __getattr__(self, name):
#              return f"Attribute {name} doesn't exist."
# 
#      obj = MyClass()
#      print(obj.some_attribute)  # Output: Attribute some_attribute doesn't exist.
#      ```
# 
#    - Note: `__getattr__` is only called when the requested attribute is not found via the usual attribute lookup.
# 
# 2. **`__getattribute__` Method**:
# 
#    - Purpose: `__getattribute__` is called whenever an attribute is accessed on an object, regardless of whether the attribute exists or not.
#    - Use Case: It's less commonly used because it intercepts all attribute access attempts, and modifying its behavior can be tricky.
#    - Behavior: This method takes two arguments, `self` (the instance) and the name of the attribute being accessed. You can use it to customize attribute access for all attributes of an object. Be cautious when implementing it to avoid unintentional infinite recursion.
#    - Example:
# 
#      ```python
#      class MyClass:
#          def __getattribute__(self, name):
#              print(f"Accessing attribute: {name}")
#              # This could result in infinite recursion if not careful.
#              return super().__getattribute__(name)
# 
#      obj = MyClass()
#      obj.some_attribute  # Output: Accessing attribute: some_attribute
#      ```
# 
#    - Note: Using `__getattribute__` to override attribute access for all attributes can be risky because it affects all attribute access attempts, including built-in ones like `__init__` or `super().__init__`. It's often safer to use `__getattr__` for custom behavior with specific attributes.
# 
# In summary, the key difference between `__getattr__` and `__getattribute__` is that `__getattr__` is used to customize the behavior when accessing undefined attributes, while `__getattribute__` is used to intercept and potentially customize all attribute access attempts, including those for existing attributes. It's important to use these methods judiciously and with caution, especially when working with `__getattribute__`, to avoid unexpected behavior or infinite recursion.

# In[ ]:


get_ipython().run_line_magic('pinfo', 'descriptors')


# Properties and descriptors are both mechanisms in Python for controlling attribute access and modification, but they differ in terms of their implementation and use cases.
# 
# **Properties**:
# 
# 1. **Implementation**: Properties are implemented using the `property` built-in function and decorators such as `@property`, `@attribute_name.setter`, and `@attribute_name.deleter`.
# 
# 2. **Use Case**: Properties are typically used to control access to, and in some cases, modification of, attributes of an object. They are primarily used to add custom behavior when getting or setting attribute values.
# 
# 3. **Scope**: Properties are defined at the class level and are associated with a specific attribute of an instance.
# 
# 4. **Simplicity**: Properties are relatively simple to use and are a Pythonic way to implement getter and setter methods without explicitly calling methods.
# 
# **Descriptors**:
# 
# 1. **Implementation**: Descriptors are implemented by defining a class that includes at least one of the following methods: `__get__`, `__set__`, or `__delete__`. This class is then used as a descriptor by defining it as a class attribute of another class.
# 
# 2. **Use Case**: Descriptors provide a more flexible and general mechanism for customizing attribute access. They can be used to control access to multiple attributes or even to implement attributes that are not directly stored in an instance's dictionary.
# 
# 3. **Scope**: Descriptors are defined as separate classes and can be associated with multiple attributes in different classes.
# 
# 4. **Complexity**: Descriptors are more complex to implement and use compared to properties. They are a lower-level mechanism that allows for more fine-grained control over attribute access.
# 
# **Comparison**:
# 
# - Properties are a simpler way to add custom behavior to attribute access and modification, especially when you only need to control one or a few attributes of an object.
# 
# - Descriptors provide a more powerful mechanism for customizing attribute access and can be used to implement advanced behaviors, such as lazy loading, data validation, and attribute redirection.
# 
# - Properties are commonly used for simple cases where custom getter and setter methods are needed.
# 
# - Descriptors are more suitable for complex cases where attribute access control needs to be applied to multiple attributes or when you want to encapsulate attribute behavior in a separate reusable class.
# 
# In summary, properties and descriptors are both used to control attribute access and modification, but properties are simpler and more commonly used for basic cases, while descriptors provide greater flexibility and are better suited for more advanced scenarios. The choice between them depends on the specific requirements of your code.

# In[ ]:


Q3. What are the key differences in functionality between __getattr__ and __getattribute__, as well as
get_ipython().run_line_magic('pinfo', 'descriptors')


# The key differences in functionality between `__getattr__`, `__getattribute__`, properties, and descriptors revolve around when they are called and how they are used:
# 
# **`__getattr__`**:
# 
# 1. **When It's Called**:
#    - Called when an attempt is made to access an attribute that doesn't exist in the object.
#    - Acts as a fallback mechanism when a requested attribute is not found via the usual attribute lookup.
# 
# 2. **Use Case**:
#    - Used for custom behavior when an undefined attribute is accessed.
#    - Typically used to dynamically compute or generate attribute values.
# 
# 3. **Scope**:
#    - Defined at the class level and is associated with specific attributes of an instance.
#    - Called only for attributes that are not found in the instance's dictionary.
# 
# 4. **Common Use**:
#    - Commonly used to implement computed properties or provide default values for missing attributes.
# 
# **`__getattribute__`**:
# 
# 1. **When It's Called**:
#    - Called whenever an attribute is accessed on an object, regardless of whether the attribute exists or not.
#    - Intercept all attribute access attempts.
# 
# 2. **Use Case**:
#    - Used for customizing attribute access for all attributes of an object.
#    - Offers a high degree of control over attribute access, but must be used carefully to avoid infinite recursion.
# 
# 3. **Scope**:
#    - Defined at the class level but affects all attributes of an instance.
# 
# 4. **Common Use**:
#    - Less commonly used due to its broad impact on attribute access. Typically used when advanced customization is required for all attribute access.
# 
# **Properties**:
# 
# 1. **When They're Called**:
#    - Called when accessing or modifying a specific attribute of an object, for which a property has been defined.
#    - Getter, setter, and deleter methods are invoked explicitly when getting, setting, or deleting the associated attribute.
# 
# 2. **Use Case**:
#    - Used to add custom behavior when getting, setting, or deleting specific attributes.
#    - Offer a convenient way to encapsulate attribute access behavior.
# 
# 3. **Scope**:
#    - Defined at the class level and associated with specific attributes of an instance.
# 
# 4. **Common Use**:
#    - Commonly used for adding getter and setter methods for specific attributes.
#    - Used for controlling attribute access for a limited set of attributes.
# 
# **Descriptors**:
# 
# 1. **When They're Called**:
#    - Called when accessing, modifying, or deleting attributes that have been designated as descriptors.
#    - Descriptor methods (`__get__`, `__set__`, `__delete__`) are invoked explicitly when accessing, setting, or deleting descriptor attributes.
# 
# 2. **Use Case**:
#    - Used to provide fine-grained control over attribute access and modification.
#    - Can be associated with multiple attributes across different classes.
# 
# 3. **Scope**:
#    - Defined as separate classes and can be associated with multiple attributes in different classes.
# 
# 4. **Common Use**:
#    - Commonly used for implementing advanced attribute behavior, such as lazy loading, validation, or attribute redirection.
#    - Ideal for controlling access to multiple attributes with shared behavior.
# 
# In summary, the key differences in functionality between `__getattr__`, `__getattribute__`, properties, and descriptors revolve around when they are called, their scope, and their common use cases. The choice of which mechanism to use depends on the specific requirements of your code and the level of control you need over attribute access and modification.
