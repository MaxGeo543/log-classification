from typing import Callable, Dict, List, Generic, TypeVar

# defines the class "Classes" which contains the classes to classify into 
# as well as an optional annotation function
# 
# Below that, an object is instanciated for the classes "Normal", "UnobservedException", "DatabaseError" and "HliSessionError" 
# The annotation strategy is very naive, and a more useful alternative should be found later...

T = TypeVar("T")
O = TypeVar("O")

class Classes(Generic[T, O]):
    """
    Defines the values a classifier should classify into and the method for annotating data
    """
    def __init__(self, 
                 classes: List[T], 
                 annotate: Callable[[O], T] | None = None
                 ):
        """
        Create a new `Classes` object

        :param classes: a list of all possible values the classifier can classify into
        :param annotate: a callable object that takes an object and returns a label (class), used to annotate objects, or `None`
        """
        self.values = sorted(classes)
        self._annotate = annotate
    
    def annotate(self, obj: O) -> T:
        """
        Annotate an object. Only works if annotate has been set in `__init__`

        :param obj: The object to classify
        :returns: The label for the passed object
        """
        if self._annotate is None: raise Exception("Cannot annotate data if _annotate is None")
        
        label = self._annotate(obj)
        if label not in self.values: raise Exception("Invalid class.")

        return label


# defines the classes. The variable "classes" will be imported by other scripts in this project

def annotate(window: List[Dict[str, str]]):
    last_event = window[-1]
    ############################
    # Annotation rules
    ############################
    # Rule for UnobservedException
    if last_event["function"] == "C_line_Control_Server.CCServerAppContext.TaskSchedulerUnobservedTaskException":
        return "UnobservedException"
    
    elif last_event["log_level"] == "Error":
        # Rule for DatabaseError
        if "DBProxyMySQL" in last_event["function"] or "DBManager" in last_event["function"]:
            return "DatabaseError"
    
            
        # Rule for HliSessionError
        elif "SessionFactory.OpenSession" in last_event["function"]:
            return "HliSessionError"
        
        else:
            return "Normal"
            
    # Rule for Normal data
    else:
        return "Normal"
classes = Classes(["Normal", "UnobservedException", "DatabaseError", "HliSessionError"], annotate)