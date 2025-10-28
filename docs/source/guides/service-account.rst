Out-of-Cluster Usage with Service Accounts
===========================================

This guide explains how to use the Kubeflow SDK (TrainerClient and KatibClient) from 
outside a Kubernetes cluster using Service Account tokens.

Prerequisites
-------------

- Kubernetes cluster with Kubeflow installed
- Permissions to create Service Accounts
- ``kubectl`` configured to access your cluster

Setup Steps
-----------

1. **Create Service Account**

.. code-block:: bash

    kubectl create serviceaccount kubeflow-sdk-user -n kubeflow
    kubectl create clusterrolebinding kubeflow-sdk-user-binding \
      --clusterrole=kubeflow-edit \
      --serviceaccount=kubeflow:kubeflow-sdk-user

2. **Get Service Account Token**

.. code-block:: bash

    # For Kubernetes 1.24+
    kubectl create token kubeflow-sdk-user -n kubeflow --duration=8760h

3. **Get API Server URL**

.. code-block:: bash

    kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}'

Using TrainerClient
-------------------

.. code-block:: python

    import os
    from kubernetes import client
    from kubeflow.trainer import TrainerClient
    from kubeflow.trainer.backends.kubernetes.types import KubernetesBackendConfig

    configuration = client.Configuration()
    configuration.host = os.getenv("K8S_API_SERVER")
    configuration.api_key = {
        "authorization": f"Bearer {os.getenv('K8S_SERVICE_ACCOUNT_TOKEN')}"
    }
    configuration.verify_ssl = True

    backend_config = KubernetesBackendConfig(
        client_configuration=configuration,
        namespace="kubeflow"
    )

    trainer_client = TrainerClient(backend_config=backend_config)

Using KatibClient
-----------------

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

    import os
    from kubernetes import client
    from kubeflow.katib import KatibClient

    configuration = client.Configuration()
    configuration.host = os.getenv("K8S_API_SERVER")
    configuration.api_key = {
        "authorization": f"Bearer {os.getenv('K8S_SERVICE_ACCOUNT_TOKEN')}"
    }
    configuration.verify_ssl = True

    katib_client = KatibClient(client_configuration=configuration)

    # List experiments
    experiments = katib_client.list_experiments(namespace="kubeflow")
    print(f"Found {len(experiments)} experiments")

Complete Katib Example
~~~~~~~~~~~~~~~~~~~~~~

Create and manage a Katib experiment from outside the cluster:

.. code-block:: python

    import os
    from kubernetes import client
    from kubeflow.katib import KatibClient
    from kubeflow.katib import V1beta1Experiment, V1beta1AlgorithmSpec
    from kubeflow.katib import V1beta1ObjectiveSpec, V1beta1ParameterSpec
    from kubeflow.katib import V1beta1ExperimentSpec, V1beta1FeasibleSpace

    configuration = client.Configuration()
    configuration.host = os.getenv("K8S_API_SERVER")
    configuration.api_key = {
        "authorization": f"Bearer {os.getenv('K8S_SERVICE_ACCOUNT_TOKEN')}"
    }
    configuration.verify_ssl = True

    katib_client = KatibClient(client_configuration=configuration)

    # Create experiment
    experiment = V1beta1Experiment(
        api_version="kubeflow.org/v1beta1",
        kind="Experiment",
        metadata=client.V1ObjectMeta(name="random-example", namespace="kubeflow"),
        spec=V1beta1ExperimentSpec(
            max_trial_count=3,
            algorithm=V1beta1AlgorithmSpec(algorithm_name="random"),
            objective=V1beta1ObjectiveSpec(
                type="maximize",
                objective_metric_name="accuracy"
            ),
            parameters=[
                V1beta1ParameterSpec(
                    name="lr",
                    parameter_type="double",
                    feasible_space=V1beta1FeasibleSpace(min="0.01", max="0.1")
                )
            ]
        )
    )

    katib_client.create_experiment(experiment, namespace="kubeflow")
    
    # Get optimal hyperparameters
    optimal_hp = katib_client.get_optimal_hyperparameters("random-example")
    print(f"Optimal HP: {optimal_hp}")

SSL Certificate Handling
------------------------

For self-signed certificates:

.. code-block:: python

    # Option 1: Provide CA certificate (recommended for production)
    configuration.ssl_ca_cert = "/path/to/ca.crt"
    
    # Option 2: Disable verification (testing only)
    configuration.verify_ssl = False

Extract CA certificate:

.. code-block:: bash

    kubectl config view --raw -o jsonpath='{.clusters[0].cluster.certificate-authority-data}' | base64 -d > ca.crt

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Connection Errors**

- Verify API server URL: ``kubectl config view --minify``
- Check network connectivity
- Ensure token is valid and not expired

**Permission Errors (401/403)**

.. code-block:: bash

    # Verify RBAC permissions
    kubectl auth can-i --list --as=system:serviceaccount:kubeflow:kubeflow-sdk-user -n kubeflow

**SSL Certificate Errors**

- Use ``configuration.ssl_ca_cert = "/path/to/ca.crt"`` for self-signed certs
- For testing only: ``configuration.verify_ssl = False``

**Debug Logging**

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.DEBUG)

References
----------

- `Kubernetes Service Accounts <https://kubernetes.io/docs/tasks/configure-pod-container/configure-service-account/>`_
- `Python Kubernetes Client <https://github.com/kubernetes-client/python>`_
- `Katib Documentation <https://www.kubeflow.org/docs/components/katib/>`_
- `KatibClient Issue #2046 <https://github.com/kubeflow/katib/issues/2046#issuecomment-1691659428>`_
