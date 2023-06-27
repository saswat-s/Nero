from owlready2 import *

class Hermit_KnowledgeBase:
    def __init__(self, url):
        self.url = url
        self.ontology = get_ontology(url).load()
        self.sync_reasoner()

    def sync_reasoner(self):
        sync_reasoner()

    def get_individuals(self):
        individuals = []
        for ind in self.ontology.individuals():
            individuals.append(ind.name)
        return individuals

    def get_all_atomic_concepts(self):
        atomic_classes = []
        for cls in self.ontology.classes():
            if owl.Thing in cls.is_a and not cls.subclasses():
                atomic_classes.append(cls.name)
        return atomic_classes

    def get_properties(self):
        properties = []
        for prop in self.ontology.properties():
            properties.append(prop.name)
        return properties

# Create an instance of the Hermit_KnowledgeBase class
kb = Hermit_KnowledgeBase(url='/home/shrushti/Nero/dllearner-1.4.0/examples/family-benchmark/family-benchmark_rich_background.owl')

# Retrieve all individuals from the knowledge base
all_individuals = kb.get_individuals()
print("All Individuals:")
print(len(all_individuals))

# Retrieve all atomic concepts from the knowledge base
atomic_classes = kb.get_all_atomic_concepts()
print("Atomic Concepts:")
print(len(atomic_classes))

# Retrieve all properties from the knowledge base
properties = kb.get_properties()
print("Properties:")
print(len(properties))
