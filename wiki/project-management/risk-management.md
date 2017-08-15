---
title: Risk Management
---
Risk management is an important process, but can easily turn into an exercise in making a list of items that people think about once and never look at again. There can be little accountability and - if only listed by name with a nebulous category - no real grasp of the risks' impact or what should be done about them.

A lot of this comes down to a lack of decent tools and of using those tools consistently. In standard risk management, risks can be split up into probability and severity. Also helpful to track are the dates on which risks are first identified, what part of a project they impact, their actual impact(an expansion on severity from a letter code to what actually could happen), what mitigation has been taken, and what the risk is after mitigation.

Probabilities and severities are usually tracked by numbers or letters. In my implementation I followed the GOV standard of probability being a number and severity being a letter for ease of understanding though unlike GOV, I made both categories increase with their numbers rather than having severity be an inverse. I personally find it easier to understand a 5A or a 3C - especially when said out loud. Changing it to make both categories ascend also feels more intuitive and is much easier to implement in a spreadsheet!

## Probabilities
The probabilities are:
1. **Rare** - The outside limit of possibility. Something usually only noted if it has a higher level of severity and so needs to be evaluated despite being an outside chance.
2. **Unlikely** - An event that with all likelihood won't happen but under poor circumstances might occur.
3. **Possible** - Might happen, might not.
4. **Likely** - Even under favorable circumstances still has a decent chance of occurring.
5. **Nearly Certain** - It's going to happen unless the team gets extremely lucky or mitigates.

## Severities
The severities are:
- **A. Negligible** - Even if it happens, the impact to the project or system is a very minor concern. At higher likelihoods might be worth mitigating just to remove an easily avoided nuisance.
- **B. Low** - A severity where the degradation of the system might be effect something the user needs or the cost/schedule overrun for the project represents a quantity that will make other development more difficult.
- **C. Moderate** - Significant degradation of a subsystem that could be as bad as 50% loss of a function on the system or an overrun that threatens to disrupt the rest of the process.
- **D. Severe** - An entire function being disabled or an overrun that could send the project significantly over budget or time
- **E. Catastrophic** - Destruction of the system or cancellation of the project if this risk occurs.

These probabilities and severities can sometimes be named somewhat differently, but the basic idea is the same and rarely does a matrix have more than five of either. If a high likelihood and a high risk coincide, that is a suggestion that the project or product itself may be untenable. For instance, identifying a 5E at the start of a project might mean it's time to find a new project.

## Actions
Combinations of probabilities and severities create an action. As with the previous, there are five broad categories of action:

- **No action:** A risk that was identified but then either adjudicated or mitigated as not worth further consideration and is safe to ignore until all higher level actions have been cleared or just ignored completely until the risk changes due to a re-evaluation based on it actually happening (vaulting a 1 to a 5 in development for instance, though in field operations it may not change the probability at all) or having more impact than expected.
- **Monitor:** A risk that is on the edge. Monitor means that the risk could cross the line into being a problem that needs to be addressed or to become a No Action and no longer a concern. Monitors should be addressed once all higher level actions have been cleared as a matter of caution.
- **Action:** Needs to be addressed or mitigated. These risks comprise the group on the line that divides your risks in half, where p = -s or high probability, low severity and low probability, high severity with the moderates of both in the middle. These are generally the most common risks being mitigated at any given time, as they're the ones that even the best process will generate.
- **Urgent Action:** Urgent actions may seem very rare given that only two states can produce them, but possible catastrophic and nearly certain moderate are two combinations that come up when spending time investigating a process or product. The first is the "Yeah, that happens a lot and is such a problem." (ex. MS Vista crashing). And the "Can it happen? Yeah... We'd better fix that." (ex .Any given house in a neighborhood being struck by lightning).
- **Immediate Action:** A drop-everything problem. Only happens when a high probability meets a high severity. Immediate because the risk means that not mitigating means the project or product will longer satisfy requirements. In other words, "If we don't fix this, the project is over." or "If we don't fix this, no one will buy our product." (ex. GM knew their ignition switches were dangerous and did not recall, an Immediate Action that was not mitigated).

> As suggested in some of the actions, the idea is to start with the worst and work backwards. Don't expect to clear your matrix entirely, but if all you have are Monitors and No Actions, then you're in very good shape.

Tracking the date increases accountability. When a project knows how long a risk has been on the books without mitigation helps them understand their process and how tough a risk is. It also helps a group feel like the risk needs to be worked on because it's been there for that long. There is also the possibility that the risk will be left on the chart for a long time and the team has reconsidered or forgotten about it, so it's good to know what phase the project was in when the risk was identified.

**Mitigating risks** is a topic of its own. The mitigation is often associated with the action, as mitigating a risk can go as far as eliminating or removing the risk, though they are not all the same thing. For the purposes of the MRSD project, consider them basically the same thing and track what was done about a risk. These actions can reduce risks to lower categories or remove them entirely from the sheet, though archiving risks that have been mitigated or eliminated is a good morale boost.

**Impact Type/Location** is what kind of risk it is (schedule, cost, physical, cybersecurity, etc), and what phase, subsystem, or operational requirement it affects. These can be tracked as separate categories if there's a lot to say, and can even be linked to the WBS/SSS or Gantt chart to show where risks are showing up. The tools to do so are a bit more involved and I haven't implemented them myself yet.

Below is a risk matrix and tracker made with some of the risks identified early in a Carnegie Mellon MRSD Project. It will take risks and display their overall values automatically in the second sheet, and output an aggregate risk matrix to the first sheet. After-mitigation risk level is not implemented, but can be kept in the comments section, and the original risk updated once mitigation has been implemented with a note as to how the mitigation affected the risk.
- [MRSD Risk Matrix & Tracker](assets/MRSD_Risk.xlsx)
