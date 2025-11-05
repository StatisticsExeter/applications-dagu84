def sum_list(numbers):
    """Return the sum of all elements in the list 'numbers'."""
    return sum(numbers)


def first_of_tuple(t):
    """Return the first element of the tuple 't'."""
    output = t[0]
    return output


def has_key(d, key):
    """Return True if 'key' exists in dictionary 'd', else False."""
    if key in d:
      return True
    else:
      return False


def round_float(f):
    """Round the float 'f' to 2 decimal places."""
    output = round(f, 2)
    return output


def reverse_list(lst):
    """Return a new list that is the reverse of 'lst'."""
    return lst[::-1]


def count_occurrences(lst, item):
    """For a list of items 'lst', count how many times element 'item' occurs."""
    output = 0
    for i in lst:
      if i == item:
        output += 1
      else:
        None
    return output


def tuples_to_dict(pairs):
    """Convert a list of (key, value) tuples 'pairs' into a dictionary."""
    output = {}
    for key, value in pairs:
      output[key] = value
    return output


def string_length(s):
    """Return the number of characters in string 's'."""
    output = len(s)
    return output


def unique_elements(lst):
    """Return a list of unique elements from 'lst'."""
    output = []
    for i in lst:
      if i not in output:
        output.append(i)
    return output


def swap_dict(d):
    """Return a new dictionary with keys and values of 'd' swapped."""
    output = {}
    for key, value in d.items():
      output[value] = key
    return output
