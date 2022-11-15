## How could you use AI for a problem that interests you? 

- What is the task?
- What kind of data would you use?
- What kind of method or model might be appropriate?
- What kind of metric would you use to measure success?

One relevant use of AI for me would be to determine whether or not someone is currently having a seizure. A more difficult problem to solve would be determining if someone is about to have a seizure sometime in the future. But, a good start would be accurately detecting seizures as they are occurring. This type of problem interests me because I did an independent study in computational neuroscience where I was trying to simulate synchronized neurons. During the study, I learned that seizures happen when the neurons in the brain become highly synchronized and repeatedly activate at the same time. Therefore, the program I wrote for this study could possibly be used to create testing data for an AI learning to detect seizures.

As for the implementation of this seizure detection model, it would likely use supervised learning and logistic regression to determine whether or not a seizure is occurring at each time step. The training data could be brain waves or some kind of neural activity. Each of these data points would need to be labeled indicating if a seizure is currently happening. The model would then learn from this data to perform some type of complex logistic regression where it attempts to categorize each data point as either seizure or not seizure. In the end, the accuracy of the model could be calculated by how many of the predicted data points' classifications matched the labeled data. 