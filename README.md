# CONCERT Extensions: Uncovering Biological Mechanisms in Spatial Transcriptomics

CONCERT is an advanced generative model built for spatial transcriptomics data that includes targeted genetic or chemical perturbations. It evaluates its predictions using three paired datasets where every measurement spot contains an exact physical tissue coordinate and a full transcriptomic readout. These include the Perturb map dataset containing mouse lung tumors with targeted CRISPR knockouts, a mouse colitis dataset measuring gut tissue at different days of induced inflammation, and an ischemic stroke dataset mapping mouse brain tissue at varying distances from a focal injury. 

Using this data, CONCERT accomplishes three primary tasks. It predicts how local perturbations alter the gene expression of nearby unperturbed cells, models how these signals propagate across sharp tissue boundaries, and simulates counterfactual scenarios by predicting how a specific genetic edit would behave if it occurred in an entirely different spatial microenvironment. This is highly interesting because standard computational tools treat cells in isolation, whereas CONCERT proves that a cellular response depends heavily on its physical location. However, it functions strictly as a predictive engine and does not attempt to explain the underlying biological mechanisms driving these expression changes.

## Proposed Research Directions

We propose two distinct research directions to uncover these missing biological mechanisms.

### Direction 1: Spatial Epistasis
Current perturbation datasets contain experiments where multiple genes are silenced simultaneously, but existing models embed these combinations simply as flat categorical variables. We propose building a model to determine if the interaction between two mutated genes changes as a function of their spatial microenvironment. Two genes might act completely independently in the center of a tumor but exhibit severe synergistic effects at the tumor boundary. By mathematically separating the baseline effect of each individual mutation from their combined interaction effect, we can continuously map how genetic dependencies physically shift across different tissue architectures.

### Direction 2: Spatially Dynamic Gene Regulatory Networks
The sequencing platform used in the Perturb map dataset captures the whole transcriptome, providing thousands of measured genes per spot and offering more than enough dimensions to model complex regulatory networks. Existing spatial network tools cannot accomplish this because they rely entirely on observational correlation, looking at a static tissue slice and drawing connections between genes that happen to activate together. Because correlation does not equal causation, mapping a true causal network requires a model that mathematically incorporates the physical CRISPR edits as explicit interventional anchors. Making definitive causal claims from spatial data requires rigorous theoretical guarantees, conceptually similar to the latent space frameworks discussed in papers like STACI. 

We propose developing a variational inference framework that maps the directed downstream effects of a perturbation, treating the network connections between genes as continuous functions of their spatial coordinates. Instead of just predicting the final transcriptome, this approach would reveal the exact causal cascade triggered by an intervention and demonstrate how the tissue microenvironment physically rewires internal cellular signaling.

***

## Project 2A: Spatial Epistasis

**Overall Approach**  
We will quantify how the genetic interaction between two simultaneous CRISPR knockouts shifts across different tissue microenvironments. By developing a probabilistic model to separate baseline mutation effects from spatial interaction terms, we will map continuous changes in genetic dependencies across the tissue architecture.

### Milestone 1: Data Preparation and Baseline Evaluation (Week 1)
- Task 1: Download and preprocess the double knockout Perturb map datasets.
- Task 2: Filter for highly variable genes and define spatial neighborhood boundaries.
- Task 3: Run a standard additive baseline model to establish initial performance metrics.

### Milestone 2: Model Architecture and Prototyping (Week 2)
- Task 1: Draft the mathematical framework separating single and joint perturbation effects.
- Task 2: Code a preliminary variational inference model using PyTorch.
- Task 3: Test the initial model on a small subset of the spatial data to check for convergence.

### Milestone 3: Full Training and Spatial Mapping (Week 3)
- Task 1: Train the model on the high performance computing cluster using the full dataset.
- Task 2: Extract the learned spatial interaction parameters from the latent space.
- Task 3: Plot the interaction strength across the physical coordinates of the tumor.

### Milestone 4: Analysis and Presentation (Week 4)
- Task 1: Identify specific genes that show the most extreme spatial epistasis.
- Task 2: Compare model predictions against baseline additive assumptions.
- Task 3: Compile the final figures and draft the summary report for the class.

***

## Project 2B: Spatially Dynamic Gene Regulatory Networks

**Overall Approach**  
We will infer how causal gene regulatory networks dynamically rewire across spatial coordinates by treating CRISPR edits as targeted interventional anchors. We will design a variational inference framework that maps directed downstream effects, establishing network connections as continuous functions of the spatial microenvironment. It is gene wise modeling. In this model, the nodes are individual genes, and we are tracking how one gene directly regulates another gene inside a cell. The spatial component simply means that this specific gene to gene wiring changes dynamically based on where the cell physically sits in the tissue.

### Milestone 1: Causal Framework Design and Data Setup (Week 1)
- Task 1: Isolate the single knockout Perturb map data and relevant spatial coordinate matrices.
- Task 2: Review theoretical guarantees from latent space literature to structure the causal graph.
- Task 3: Formulate the probabilistic objective function linking spatial coordinates to network edges.

### Milestone 2: Prototype Spatially Aware Network Model (Week 2)
- Task 1: Implement the generative model structure with spatial kernels or continuous graph embeddings.
- Task 2: Integrate the physical CRISPR perturbation as a strict causal node in the framework.
- Task 3: Debug the loss function on a highly reduced data patch.

### Milestone 3: Cluster Execution and Network Inference (Week 3)
- Task 1: Deploy the training scripts on the cluster for the full tissue slide.
- Task 2: Extract the predicted directed edge weights for thousands of genes.
- Task 3: Validate that the inferred downstream effects align with known biological pathways.

### Milestone 4: Network Visualization and Synthesis (Week 4)
- Task 1: Generate visual graphs showing how specific causal edges strengthen or weaken across the tissue.
- Task 2: Highlight key signaling cascades that diverge between the tumor core and periphery.
- Task 3: Finalize the technical presentation detailing the causal discovery results.
