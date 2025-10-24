* resnet10_organamnist_avg_pool.pth：The ResNet-10 network trained using Medminist's Organamiminist has a Test Loss of 0.5395 and a Test Acc of 0.8789

* resnet-10.py：Used for training the ResNet-10 model on the Medminist dataset，plaintext training

* trans_to_he_friendly.py：Used to transform the trained ResNet-10 model layer by layer for homomorphic encryption input inference，Ciphertext inference

* trans_to_he_friendly_tenseal.py：trans_to_he_friendly.py but use tenseal

* utils.py：used to test CKKS parameters
