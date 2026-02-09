# Prompt

I want to try to train a neural net that would be able to grade a moonboard boulder using the font scale. I have an example of the training data in @example.json . The important properties are "grade" containing the font scale grade, and the moves array. The moves array contains the holds which are included in the boulder. The important properties here are "description", containing the hold location, "isStart" which is true if it is a starting hold, and  "isEnd" which is true if this is an ending hold. The "description" pinpoints the location of the hold on a 2D grid, so it contains values such as "F7", "B9", "A11". The colums are from A to K, and the rows are from 1 to 18, 1 being the bottom most row.

The resulting trained neural net should be able to accept an array of all the moves in the boulder, and output a font grade determining its difficulty.

While I have a lot of experience in software engineering, I don't have any practical experience in neural nets, so I would need a lot of guidance in which tech stack I should use, which sort of neural net to train, and putting it all together. 

I found a paper on this subject, but I'm not sure if it's helpful - it's in the moonboard paper pdf file. Analyse it and check if it has information helpful to successfuly implementing this.

What I you to do is to create a step-by-step incremental plan on how to implement this starting from scratch. If you think more information would be helpful, ask me claryfing questions one at a time.
