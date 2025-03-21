# **Technical Case for Machine Learning Engineer**
### **OBJECTIVE**
The goal of this challenge is to assess your ability to develop\, structure\, and operationalize a Machine Learning pipeline in a production environment\. We want to understand how you build scalable systems\, implement software engineering and MLOps best practices\, and ensure efficiency in model inference and maintenance\.
More than the model itself\, we are interested in your engineering capabilities\, code organization\, and your approach to deployment\, monitoring\, and scalability\.

### **CHALLENGE**
Your challenge is to build a service that exposes a RESTful endpoint for making predictions using a Machine Learning model\. To achieve this\, you should\:
* Create an API that receives input data and returns predictions\.
* Develop an optimized pipeline for inference in production\.
* Implement a deployment method to ensure that the solution can be used in a scalable way\.
* Add logging and monitoring to track the performance of the model and API\.
The model should be a solution to the classic **Titanic survival prediction challenge on Kaggle**\. This is a well\-known problem with many publicly available solutions\. The focus is more on **engineering** than on the model itself\.

### **EVALUATION CRITERIA**
This case aims to understand how you handle real\-world **Machine Learning Engineering** challenges\, including\:
1. **Back\-end development\:** Structuring an API to serve the model\.
2. **Infrastructure and MLOps\:** Containerization and deployment strategies\, training\, and retraining\.
3. **Scalability\:** Efficiency of the inference pipeline and training process\.
4. **Modularity\:** How the code is organized into components and functionalities\, making it usable by a Data Scientist\.
5. **Monitoring\:** Logging\, metrics\, and tracking strategies\.
6. **Production\-readiness\:** Ensuring the code meets the requirements of a real\-world Machine Learning application\.
7. **Code best practices\:** Organization\, design patterns\, logging\, and documentation\.

### **TECHNICAL REQUIREMENTS**
For the implementation\, you can use the tools of your choice\, but we suggest the following\:
**Programming Language\:** Python or Go
* **API Framework\:** Your choice
* **Modeling\:** Scikit\-learn\, TensorFlow\, PyTorch\, or LightGBM \(simple model\) – model performance will **not** be a grading criterion
* **Deployment\:** Docker \(Kubernetes or another scalable approach is a plus\)\, Terraform\/IACs\, and CI\/CD
* **Feature Store\:** Repository for automated training and continuous model improvement
* **Model Registry\:** MLflow\, Kubeflow\, or another of your choice
* **Observability\:** Logging and monitoring strategy \(e\.g\.\, Prometheus\, ELK\, or another solution\)
* **Delivery\:** GitHub repository with a **README** explaining your solution and instructions on how to reproduce and test its production behavior\.
