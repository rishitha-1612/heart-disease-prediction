class SharedMemory:
    """
    Stores outputs from all modules
    """

    def __init__(self):
        self.memory = {}

    def store(self, module_name, data):
        """
        Store output from a module
        """
        self.memory[module_name] = data

    def get(self, module_name):
        """
        Retrieve output of a module
        """
        return self.memory.get(module_name, None)

    def get_all(self):
        """
        Retrieve all module outputs
        """
        return self.memory

# example usage
if __name__ == "__main__":
    memory = SharedMemory()

    memory.store("echo", {"label": "Abnormal", "confidence": 0.8})
    memory.store("ecg", {"label": "Arrhythmia", "confidence": 0.7})
    memory.store("clinical", {"label": "High Risk", "confidence": 0.9})

    print(memory.get("echo"))
    print(memory.get_all())


# Later in main.py, flow will be:

# Echo → store in memory  
# ECG → store in memory  
# Clinical → store in memory  

# Fusion → read from memory → calculate risk