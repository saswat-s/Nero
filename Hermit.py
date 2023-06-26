from rdflib import Graph, RDF, OWL

class Hermit_KnowledgeBase:
    def __init__(self, url):

        
        # Load the RDF/XML file
        self.graph = Graph()
        self.graph.parse(url)
        
    def get_individuals(self):
        individuals = []
        for subject in self.graph.subjects():
            individuals.append(subject)
        return individuals
    
    def get_all_atomic_concepts(self):
        atomic_concepts = []
        for subject in self.graph.subjects(predicate=RDF.type, object=OWL.Class):
            atomic_concepts.append(subject)
        return atomic_concepts
    
    def get_all_atomic_properties(self):
        properties = []
        for subject in self.graph.subjects(predicate=RDF.type, object=OWL.ObjectProperty):
            properties.append(subject)
        return properties


# Create a Hermit_KnowledgeBase instance and load the OWL file from localhost
file = Hermit_KnowledgeBase("./KGs/Family/Family.owl")
# file = Hermit_KnowledgeBase(url="http://localhost:8080/hermit")

# Retrieve individuals
all_individuals = file.get_individuals()
print("All Individuals:")
print(len(all_individuals))

# Retrieve atomic concepts
atomic_concepts = file.get_all_atomic_concepts()
print("Atomic Concepts:")
print(len(atomic_concepts))

# Retrieve properties
properties = file.get_all_atomic_properties()
print("Properties:")
print(len(properties))
