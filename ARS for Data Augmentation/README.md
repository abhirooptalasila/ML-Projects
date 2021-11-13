# Augmented Random Search for Data Augmentation

Data augmentation policies are generally dataset-specific. Recently, Autoaugment was proposed using RL as an automatic augmentation approach. AutoAugment searches for the augmentation polices in the discrete search space, which may lead to a sub-optimal solution. In this implementation, we employ the Augmented Random Search method (ARS) to improve the performance of AutoAugment. The main change is moving from discrete search space to continuous space, which will improve the searching performance and maintain the diversities between sub-policies. With the proposed method, state-of-the-art accuracies are achieved on CIFAR-10, CIFAR-100, and ImageNet (without additional data). 





## References

1. https://arxiv.org/abs/1811.04768
2. https://towardsdatascience.com/introduction-to-augmented-random-search-d8d7b55309bd
3. https://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf