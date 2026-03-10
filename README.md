[Explainable Synthetic Image Detection.pdf](https://github.com/user-attachments/files/25499464/proposal.pdf)
[slides.pptx](https://github.com/user-attachments/files/25871247/slides.pptx)

Utilizing K-means and PCA, we performed EDA no the OpenFake dataset revealing that there was significant discrepency between both content and resolution of real images and synthetic images. As illustrated in plot 1 (the style clusters), there were predominant style groupings based on image style alone. we further explored the difference between the image classes (plot 2), revealing a significant difference between real and fake image content. To evaluate whether the dataset possessed any style overlap between real and fake images, we evaluated the style space alone, revealing that cluter 0 from plot 1 contained significant style deviations from the fake image class. We then evaluated the resolution of these images, finding that a large number of real images deviated from the fake images in both style and resolution. Finally, we identified the cluster with the most overlapping content and resloution to be cluster 1.

![c0519b96-7384-4c1a-93a2-3d38e7d75463](https://github.com/user-attachments/assets/eb6c0abf-cf1f-4df1-8915-92b6b8536c31)

Using this knowledge, we selected the cluster with the most content overlap (cluster 1), obtaining a balanced real/fake image dataset with nearly 90% content overlap. we additionally, filtered the image data further based on resolution and style. This pruned significant resolution and stylistic discrepencies between classes, but provided a sufficient remaining train split of ~4,500 real/fake images. The resulting cleaned dataset was a balanced split between real and fake images based on shared content, style, and resolution. 

![50315245-c794-41be-817e-82cb14906b19](https://github.com/user-attachments/assets/cf1754b5-c8d9-4c78-9af2-5187655c4762)
