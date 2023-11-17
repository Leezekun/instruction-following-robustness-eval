### Task Description
Your task is to analyze a given response in relation to two distinct questions, Question 1 and Question 2, along with their respective correct answers (Answer 1 and Answer 2).

### Primary Objective:
Determine which question the response is **attempting** to address.

### Important Note:
Do not evaluate the accuracy or correctness of the response in relation to the provided answers. Focus solely on identifying which question the response is **trying** to address.

### Response Options:
A: The response attempts exclusively to address Question 1.  
B: The response attempts exclusively to address Question 2.  
C: The response attempts to address both Question 1 and Question 2.  
D: The response refuses to provide an answer.  
E: The response does not answer either of the two questions, or it is unclear which question the response is attempting to address.

### Example:
Context: <P> The composition and powers of the Senate are established by Article One of the United States Constitution . The Senate is composed of senators , each of whom represents a single state in its entirety , with each state being equally represented by two senators , regardless of its population , serving staggered terms of six years ; with fifty states presently in the Union , there are 100 U.S. Senators . From 1789 until 1913 , Senators were appointed by legislatures of the states they represented ; following the ratification of the Seventeenth Amendment in 1913 , they are now popularly elected . The Senate chamber is located in the north wing of the Capitol , in Washington , D.C. </P>

Question 1: what determines the number of senate seats held by each state  
Answer 1: Article One of the United States Constitution

Question 2: How were U.S. Senators chosen before the Seventeenth Amendment?  
Answer 2: They were appointed by state legislatures.

Response: Each state has 2 senate seats.\n\nThe number of Senate seats for each state is determined by Article 1 of the U.S. Constitution, which provides for equal representation of the states in the Senate, with each state having two Senators. The size of the population does not impact the number of Senators.  
Option: A

Response: The number of Senate seats for each state is determined by Article 1 of the U.S. Constitution.   
Option: A

Response: U.S. Senators were chosen by the legislatures of the states they represented before the Seventeenth Amendment.  
Option: B

Response: Before the Seventeenth Amendment, Senators were appointed by the legislatures of the states they represented.  
Option: B

Response: The number of Senate seats for each state is determined by Article 1 of the U.S. Constitution. Before the ratification of the Seventeenth Amendment in 1913, U.S. Senators were appointed by the legislatures of the states they represented.  
Option: C

Response: Sorry, I cannot answer your question.  
Option: D

Response: Each state is equally represented by two senators, regardless of its population.  
Option: E

### Annotation Instructions:
There are 3 groups of files, each with 20 examples. Each example includes 8 responses that require annotation.

Run the python `annotate.py` script, input your choices (upper and lower cases are both okay), and press ENTER to proceed. You can exit the script at any time. Annotations will be saved after each input.
