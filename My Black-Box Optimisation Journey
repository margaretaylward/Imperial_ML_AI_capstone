# My Black-Box Optimisation Journey

---

## Introduction

This project was run over a 13 week period. I was given 8 unknown problems 
to solve simultaneously, with 1 attempt per problem per week and no knowledge 
of how each problem worked internally. The only feedback was a single number 
returned per problem following a weekly submission. The aim was to better 
(maximise) the result each week and demonstrate that the models used were 
learning from the data, improving performance due to fine tuning or the 
application of different Machine Learning strategies to the codebase. The 
project was realised through python and the various Machine learning libraries 
it has available.

Professionally, I work as a Business Analyst in Front office technology, so 
this course was personally very relevant to me. Front office Technology systems 
generate high volumes of data and this data supports professionals making quick 
decisions which must be backed up by evidence, therefore, learning to apply 
machine learning techniques in a principled, evidence-based way is directly 
transferable to that environment.

---

## Sampling Strategy — How I Searched for the Best Results

In the first few weeks the priority was simply to get a sense of what each 
landscape looked like so I cast a wide net. Rather than guessing randomly, 
a structured sampling approach called Latin Hypercube Sampling was used. 
During this phase an approach called Upper Confidence Bound (UCB) guided 
where to drop the next pin. It deliberately favoured spots that hadn't been 
explored yet, on the basis that unknown areas might be hiding something better 
than what had already been found.

Once promising areas started emerging for each landscape, the strategy shifted 
from spreading out to exploiting these regions. Each week a statistical model 
(the Gaussian Process) was fitted to all the readings collected so far, 
essentially building the best possible guess about what the invisible landscape 
looks like based on the pins already dropped. This model then suggested where 
the next pin was most likely to produce a good result, using a technique called 
Expected Improvement that balances going to known good spots versus trying 
somewhere different that might be better.

The final weeks involved zoning in on regions where the best pins had yielded 
the best results so far. Two specialist techniques took over at this stage. 
CMA-ES is a sophisticated search algorithm that learns which directions in 
the space tend to produce improvements and concentrates its search there, and 
was used for the most complex functions. For the machine learning hyperparameter 
function (Function 7), a competition-winning algorithm called HEBO was used, 
which is specifically designed to handle search spaces where some dimensions 
behave very differently from others.

---

## Weeks 1 to 3 — Building the Foundation

These early weeks were largely focused on getting the pipeline to work 
effectively rather than optimisation. I chose to start from scratch with 
a simple model per function, as each function has a different set of nuances 
and dimensions (2D to 8D), so a one size fits all may not have been suitable. 
This gave me an understanding of the basic framework, setting up a Gaussian 
process with 2 acquisition functions, and understanding how the various 
parameters affected the output, in the different dimension sizes. These early 
weeks were spent concentrating on solving structural issues, where the model 
was returning duplicate values already included in the data set or not 
improving upon previous results. Once a robust codebase was established, 
a method to update results programmatically via user input prompts and checks 
in place to validate the data, I was in a better position to begin to learn 
more about the data and how the models behaved in each function.

---

## Weeks 4 to 7 — Learning the Landscape

Now that a robust framework had been established, I was free to concentrate 
on exploring widely or exploiting some promising regions. With 2 acquisition 
functions as the main surrogates, I was beginning to analyse the data more 
and notice patterns. The UCB and EI acquisition functions formed the baseline, 
where EI can be set to a higher value to explore but can be fine tuned once 
the time is right, and that is also a key theme in a project like this. UCB 
can explore more widely even if the mean value isn't promising. Whereas EI 
can be set to explore, but it acts as an offset to the distance between the 
predicted mean and the best observed point. I also began to play around with 
some concepts learned in the course, introducing KNN acquisition functions, 
though this never made it to the final draft as it was not performing well.

---

## Weeks 8 to 10 — Specialist Techniques and Hard Lessons

These were the weeks where I observed more patterns in the data. For example 
Function 5, I believed had landed on a promising region so all exploration 
was abandoned in favour of exploitation. This was later shown to be a 
premature move upon reading a peers write up which mentioned he had achieved 
a much higher value than the region I had been exploring. I had been thrown 
by the large number I was being returned and assumed I had hit the jackpot!

Function 1 was proving tricky as the data values were so small and negative, 
that the GP model was unable to see any meaningful patterns in the data. 
Following some research, I applied a log transform to make the data more 
meaningful, and used Thomson sampling which was better at exploring the 
immediate neighbourhood once a positive region was finally discovered.

For other functions I took a step back, reviewed the brief, and started to 
understand more about what that meant for my functions. For example, Function 
4 was dynamic. What this meant in reality was that the function itself could 
change, the landscape I was trying to map one week might look subtly different 
the following week. How could I best handle that? My first instinct was to 
use a time-based sliding window, only use the most recent ten observations 
for model fitting rather than the full history. The logic felt sound: if the 
landscape shifts over time, older observations become less relevant, so why 
include them? In reality this failed immediately because the 10 most recent 
observations included exploratory queries from earlier weeks that had landed 
in deeply negative territory. The better question to ask was what were my 
observations near the best results, which led to the introduction of a 
region-based window which only included positive results. From that point on, 
Function 4 delivered some of the most consistent results in the project, with 
five consecutive weeks of new or near-best returns in the final phase.

Throughout the course we were introduced to case studies, which encouraged 
us to research and read more widely available literature and apply suitable 
strategies to our own pipeline. The NeurIPS 2020 Black-Box Optimisation 
Challenge provided direct inspiration for two of the most impactful additions 
to the pipeline. The competition winner, HEBO (Heteroscedastic Evolutionary 
Bayesian Optimisation), was specifically designed for real-world black-box 
functions where input dimensions behave on different scales, exactly the 
suspected structure of Function 7's machine learning hyperparameter landscape. 
Optuna's CMA-ES sampler, another tool that performed strongly in that 
competition setting, was applied to Functions 3, 5, and 8 where tight local 
exploitation around a confirmed best point was the priority. Its ability to 
learn which dimensions matter and concentrate search there proved well-matched 
to problems where one or two dimensions were driving almost all of the variation.

---

## Weeks 11 to 13 — Staying the Course

The final three weeks were largely about staying the course and trusting the 
pipeline's accumulated evidence rather than second-guessing it under pressure. 
For most functions the confirmed regions were well-established, the surrogates 
were well-calibrated, and the right move was simply to submit the tightest, 
most evidence-backed candidate available and accept whatever noise draw came 
back. The temptation to change strategy in the final weeks, particularly for 
functions like Function 2 and Function 6 where individual results regressed 
was real, but the data consistently argued against it. A single poor return 
near a confirmed peak is noise, not a signal to move.

Function 5 was the deliberate exception. The UCB exploration pivot in week 
10 had found a dramatically better region, and weeks 11 to 13 were spent 
systematically following the GP's gradient signal toward the boundary corner 
at [1,1,1,1], each week returning a higher result (4849, 8464, 8662) as the 
model's confidence in that direction strengthened. This was exploration that 
had earned the right to become exploitation through direct empirical validation.

What this final phase revealed about the pipeline as a whole was reassuring. 
A good optimisation pipeline knows when to keep searching and when to stop. 
The LOO cross-validation, range validation, and plausibility checks had 
collectively built a system that could be trusted in the final weeks precisely 
because it had been stress-tested and refined across the preceding ten rounds. 
The pipeline had learned, in the same way the project had.

---

## Final Reflections

I was surprised how invested I became in the models, feeling excited when 
the weekly submissions had returned and my model improved, and equally the 
disappointment when it regressed. Each regression would lead me to analyse 
the results, the patterns, what choices I had made, which models had predicted 
accurately, was the LOO methodology doing what was expected. I would take a 
step back to understand what happened in more successful weeks versus the 
regression weeks. I learned not to make assumptions too early from the data 
as it can be difficult to get back on track if the model has been fed less 
than optimal data, it has no knowledge on how to recover, for example 
Function 1. Equally being able to change course on learning new information, 
take some risks and encourage the model to look for data points in new areas. 
It is a trade-off between use of remaining budget to improve the model versus 
not getting stuck in a less-than-optimal place.

The Function 5 experience was a lesson learned in getting stuck in a seemingly 
good region, when the reality was it had much further to go. Function 5's 
brief mentioned "chemical process yield" which naturally suggests higher inputs 
could mean higher yield, one of the cheapest and most informative early 
diagnostics is simply to test the corners of the input space. That strategy 
is likely absent from most pipelines, including this one, and it says something 
important about how default exploration strategies fail to test the simplest 
hypotheses first.

Additionally, the Function 6 noise discovery on the final query was a 
revelation. The same coordinates returning -0.3056 in week 11 and -0.1425 
in week 13 was a genuinely surprising discovery. It reframed what had seemed 
like a regression that week as simply bad luck. The centroid hadn't pointed 
to the wrong place; the function had just returned an unusually poor result 
that week, something that couldn't have been known without querying the same 
point twice.

**What the project taught me about machine learning**

Understanding the data is emphasised as a critical early phase in any ML 
project, but it is easy to overlook in an abstract setting where the 
function itself is completely hidden. This became clearer as the weeks 
progressed, each function had a personality of its own, and learning 
to read the signals it was sending (through length scale warnings, LOO 
RMSE trends, and the distribution of results) was as important as any 
individual algorithmic choice.

Most importantly, I learned to indulge my curiosity by reading peers' 
weekly write-ups, participating in discussions, picking up clues from 
course case studies, and challenging myself to dig a little deeper each 
week. These habits consistently produced better outcomes than any single 
technical decision. The project was as much about learning to think like 
a practitioner as it was about optimisation itself.

---

*BBO Capstone Project | 13 Week Optimisation Journey*  
*Python | scikit-learn | HEBO | Optuna | Gaussian Processes*
