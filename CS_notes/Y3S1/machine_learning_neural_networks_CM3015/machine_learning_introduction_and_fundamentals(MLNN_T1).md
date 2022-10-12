---
tags: [bounding-box, object-detection, machine-learning, face-tracking, body-tracking, driverless-cars, handwriting-recognition, speech-recognition, privacy-concerns, recommender-system, data-imputation, generative-machine-learning, sensor-based-activity-recognition, supervised-learning, unsupervised-learning, reinforcement-learning]
aliases: [intro to machine learning, MLNN T1, machine learning topic 1]
---

# Readings for this topic

1. Introduction to machine learning: [Chapter 1 up to 1.3 of Alpaydin, E. _Introduction to machine learning](https://ebookcentral.proquest.com/lib/londonww/bookshelf.action)

# Applications of machine learning

Machine learning as a branch of artificial intelligence that in essence enables machines to learn by example. It is related to field such as computer vision, signal processing, and data mining. Due to the increased availability of data, along with the increased amount of computational power available, we're seeing an increase in the use of machine learning algorithms and products that are part of our everyday lives, mobile phones, personal assistants, and so on. 

## Face Tracking and Recognition

- E-passport gates at airports. By using advances in machine learning, computer vision, and biometrics, e-passport gates use face recognition systems that identify passengers with a high probability. 
- Face detection systems him to detect the location of faces and an image. This is usually done by finding a specific part of an image where the object of interest appears called a <b>bounding box</b>. When applying such methods to other objects besides faces, we call the task <b>object detection</b>. 

## Body Tracking

Computer vision can also be used to identify human body location, posture, and even facial expressions. This can be applied to video as well as to still images, allowing us to use machines to analyze the dynamics of people's movement. Many of the applications of machine learning deal with different types of data. As we've seen, face recognition can be used to analyze visual data that has images or videos. 

## Handwriting and Speech Recognition

Machine learning can also be applied to audio and text, and intersecting with areas such as audio and natural language processing. For example, in handwritten digit recognition, consumer oriented personal assistants that can recognize speech. These can make use of machine learning to do online translation, for example. Of course, these systems are not perfect and the quality as well as the volume of data as a significant factor affecting the accuracy of modern machine learning systems.

## Privacy concerns

There's also the issue of privacy, particularly with systems that capture personal data. As data scientists, we need to be conscious of what data has been recorded, who has access to it, and have it may be used. 

## Driverless cars and its ethical concerns

Autonomous vehicles are an important focus for research in machine learning. Creating cars that can drive themselves is a complicated challenge that includes issues such as pedestrian detection and in some cases, autonomous decision-making. Millions of images of streets and pedestrians are required in order to create datasets where machine learning algorithms can be effective. A particular challenge is to detect objects and scenes that are crowded. Real-world data is often noisy and may contain occluded object or corruptions. Obvious ethical issues arise here. What if the driverless car gets it wrong and causes an accident? Who then is to blame? The person who isn't driving? The company that sold the car? The software engineers who built the system? 

## Recommender Systems

Another machine learning application that we encountered in our daily lives as recommendation algorithms. These aim to predict products that you might like using factors such as your previous shopping behavior, the items that are in your basket, or what customers with similar preferences to yours have already purchased. A common problem in this application is that there are missing data. No-one knows how you would really rate a product that you haven't reviewed or even purchased. Therefore, techniques in <b>data imputation</b> or completion are used which aim to provide a meaningful completion to a matrix that represents your preferences. Recommender systems are also used to automatically control what fits, adds and even news we see on social media. If you enjoyed watching this content, perhaps you'd be interested in watching this other similar thing that are dangerous here. If we're only ever exposed to things that are similar to what we've already seen, there's a real danger of these systems skewing or view of the world. These systems are increasingly used in political advertisements and there's evidence that they play a negative role in increasing political polarization. 

## Generative Machine Learning

Generative models in machine learning are techniques that can generate data. For example, given a sample of someone's handwriting, can we produce other text with the same handwriting? Or given examples of many faces, can we learned to generate new faces that don't exist but are similar to example faces? Generative models and deep learning, have shown promising results in this direction. Again, there are some obvious ethical issues with this technology, which is helpful for us as data scientists to be aware of. 

## Sensor-based Activity Recognition

Sensor-based wearable activity recognition has made us we enter the mainstream to rest worn devices such as the Fitbit and Apple Watch. Sensors collect diverse information on things such as body movement, eye gaze, and heart rate and use this to automatically determine what kind of state you're in, your locomotion, eg, whether you're standing, walking, or sitting, how healthy your heart is, or even what food and drink you're consuming.

# Machine Learning Types

## Supervised Learning

- The label, is associated with every sample.
- We are trying to learn the mappings from the inputs to the outputs.

## Unsupervised Learning

- We simply observe a dataset consists of all individual samples, but with no lobals given.

![[machine_learning_types_flowchart.png]]

## Reinforcement Learning

- In reinforcement learning, we are interested in predicting a sequence of actions that entail a specific reward.