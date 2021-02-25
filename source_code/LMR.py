import numpy as np

# Local Imports
import entity

class LMRContentOwner(entity.ContentOwner):

    MSBS = np.array([2,3,4,5,6,7,8])

    def __init__(self):
        super(LMRContentOwner, self).__init__()
        
        
        
class LMRDataHider(entity.DataHider):
    
    def __init__(self):
        super(LMRDataHider, self).__init__()
        
    
class LMRRecipient(entity.Recipient):
    
    def __init__(self):
        super(LMRRecipient, self).__init__()