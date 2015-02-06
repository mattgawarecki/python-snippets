def insertion_sort(input):
"""
A hot mess, slightly resembling the insertion sort algorithm.

  input: an iterable containing some number of unsorted integers
  output: a sorted list corresponding to the input
"""
    # Copy into a new list to avoid mangling the input
    output = [i for i in input]
    for ix, item in enumerate(output):
        tmp = ix - 1    # start looking at the previous list index
        
        # Keep moving backward through the list as long as the current
        # list item is less than the one we're testing against
        while tmp >= 0 and item < output[tmp]:
            tmp -= 1
        
        # remove the current list item from its original spot
        output.pop(ix)
        
        # ... and put it in the (tmp + 1) index;
        # [tmp] would be the first place where our item would violate the sorting order,
        # so we move up by one; if the item was at the correct index the whole time (i.e. it was
        # already in sort-order), then ix == tmp + 1
        output.insert(tmp + 1, item)
    return output
