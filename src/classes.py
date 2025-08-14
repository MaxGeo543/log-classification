from typing import Callable, Any
from collections import namedtuple


class Classes:
    def __init__(self, classes: list[str], annotate: Callable[[list[dict[str,str]]], Any] | None = None):
        self.values = classes
        self._annotate = annotate
    
    def annotate(self, window: list[dict[str,str]]) -> Any:
        if self._annotate is None: raise Exception("Cannot annotate data if annotate is None")
        l = self._annotate(window)
        if l not in self.values: raise Exception("Invalid class.")
        return l





def annotate(window):
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