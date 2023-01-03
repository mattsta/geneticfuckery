geneticfuckery: genetic optimization for python variables
=========================================================



## what now?

This package provides two big features:

- a way to store multiple versions of python variables transparently to your code implementation
  - (think: "revision control, but for your value switching")
- a way to combine parameter groups together into new parameter groups
  - (think: optimization / genetic algorithm breeding)

Basically, instead of having:

```python
result = variable_a + variable_b
```

You can store the values externally then read them locally as:

```python
import geneticfuckery.gf as gf

g = gf.GeneticFuckery(dbfile="myvariables.sqlite3")

result = g.variable_a + g.variable_b
```

but, uh... what good does storing values externally do?

External variable storage allows you to easily:

- create named groups of parameters for switching between them as atomic units
- record performance of parameter groups across runs to find best combinations
- run training loops over parameter groups, record the output values, then go back and fetch high/low performing combinations

The overall goal here is to _optimize_ combinations of parameters. After your program generates results using your parameters,
`geneticfuckery` can record the results of using a parameter group, then you can query the variable database for
the best performing combinations of parameters over all runs ever created.

### Data Layout

The goal is to store parameters inside parameter groups for easy recall, but we also need to record the _results_ of running
a parameter group too, so how do we name results?

### result saving

Saving the results of running a parameter group requires more metadata about _what_ you ran your parameters against.

The gf data model relies on some basic structures:

- `pid` — parameter group id shared by parameters under one name for easy launching and reporting
- `param` and `name` — a two level hierarchy defining your parameters
    - having params as a hierarchy allows you to "group" similar values together like:
        - `green: high=3 medium=2 low=1`
        - `red: high=7 medium=5 low=0`
        - then you can reference them nested: `gf.green.high, gf.green.low, gf.red.medium`
- `results` — saving results of running parameter groups enables `geneticfuckery` to combine "best runs" and automatically generate new parameter groups.
    - results are collected by saving:
        - current parameter group id
        - dataset being evaluated ("climate change data 2022-12")
        - algos executed against dataset ("predict co2, predict temp, predict arctic ice sea, predict ice sheets, predict sea level, predict ocean warming")
        - algo parameter results ("high, low, median, success, failure, ...")
    - result layout allows a single program run to output results for _multiple_ algo evaluations per run, so instead of `O(dataset * algo)` runs, you only need `O(dataset)` count runs since your program can report an unlimited number of inner algo results per actual param group + dataset execution.
    - result layout _also_ allows you to generate reports across the two (dataset x algo) dimensions like "give me the lowest sea level prediction across all runs of the same dataset" etc



## Sample usage

### First you need to load your initial variables into the database

### Naming

gf supports a single-level nested "parameter" with multiple "names" attached to the "parameter" for creating variables, but gf _also_ supports any arbitrary nesting of final names themselves too for super easy dot access.

Every double underscore in your database variable name generates a dot access, so naming your db variables:

- `seconds_15__stop_threshold__fast = 12.3` => can be accessed as `gf.seconds_15.stop_threshold.fast`
- `seconds_15__stop_threshold__slow = 7.2` => can be accessed as `gf.seconds_15.stop_threshold.slow`
- `seconds_30__stop_threshold__fast = 22.2` => can be accessed as `gf.seconds_30.stop_threshold.fast`
- `seconds_30__stop_threshold__slow = 17.9` => can be accessed as `gf.seconds_30.stop_threshold.slow`

(here, the `seconds_{15,30}` are the "parameters" while `stop_threshold__{fast,slow}` are "names" under each parameter" (yes, these choices of terms is confusing and bad but it's what we're rolling with for now))

This also means you can do _fancy_ readers using dynamic name fetching to group access easier with python shorthand (and these sub-extracted readers can be passed around for better encapsulation across program boundaries since further readers can't go "higher" than the access they are given):

```python
gf = GeneticFuckery()
def param_duration_helper(duration: int):
    return getattr(gf, f"seconds_{duration}")

d15 = param_duration_helper(15)
d30 = param_duration_helper(30)

d15fast, d15slow = d15.stop_threshold.fast, d15.stop_threshold.slow
d30fast, d30slow = d30.stop_threshold.fast, d30.stop_threshold.slow
```


### Data Types

By default, parameters are assumed to be float/real unless otherwise noted by a string suffix.

Currently supported suffix data types are:

- `_int` => `int(x)`
- `_bool` => `bool(x)`
- `choice` => `list(x)[0]`
- (anything else or no type declared) => `float(x)`
    - though, we have two special float types:
        - `_real_0` limits breeding between `[0, 1)`
        - `_real_1` limits breeding between `[1, 2)`

The datatypes serve two purposes:

- proper type retrieval from the parameter database (so if you store 3 you get back int(3) and not str(3))
    - implementation note: all values are saved as JSON to the database, so there's JSON-like type conformance we're extracting from (which helps us when storing lists or nested values).
- declaring types per-parameter allows optimization/breeding of parameters with the proper approaches for different types (breeding floats is different from ints is different from breeding bools is different from breeding "pick one from a pile" of choice categories)

Also note: the type suffix _does_ travel with your parameter name, so if your name is `hello__thanks_int = 4` you access as `gf.hello.thanks_int`.

TODO:

- add more types (?)
- perhaps add a more complete "type schema" along with extensible breeding algos per type instead of having types + extractors + mutators all hardcoded in different places (yay more metaprogramming!)

### load yor datah

You're almost read to load your data!

Just generate a text file with your initial values named the right way, run the loader, then you'll have a database ready to use.

```haskell
seconds_15__stop_threshold__fast = 12.3
seconds_15__stop_threshold__slow = 7.2
seconds_30__stop_threshold__fast = 22.2
seconds_30__stop_threshold__slow = 17.9

# more things...
overall__decay_rate_int  = 5
runtime__single_trigger_bool = 1
runtime__exit_check_choice = ["custom", "defined", "30 seconds", "on-demand"]
```

Then load them...

```haskell
~/repos/geneticfuckery$ bat loadme.js
───────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       │ File: loadme.js
───────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1   │ seconds_15__stop_threshold__fast = 12.3
   2   │ seconds_15__stop_threshold__slow = 7.2
   3   │ seconds_30__stop_threshold__fast = 22.2
   4   │ seconds_30__stop_threshold__slow = 17.9
   5   │
   6   │ # more things...
   7   │ overall__decay_rate_int  = 5
   8   │ runtime__single_trigger_bool = 1
   9   │ runtime__exit_check_choice = ["custom", "defined", "30 seconds", "on-demand"]
───────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

~/repos/geneticfuckery$ GF_DB=testing.db poetry run gf-load loadme.js
2022-12-23 13:11:01.101 | INFO     | geneticfuckery.gf:externalLoader:1224 - [testing.db :: None] Loading parameters...
2022-12-23 13:11:01.101 | INFO     | geneticfuckery.gf:externalLoader:1224 - [testing.db :: None] Loaded parameters!
2022-12-23 13:11:01.107 | INFO     | geneticfuckery.gf:populate:325 - Adding seconds_15__stop_threshold :: fast = 12.3
2022-12-23 13:11:01.107 | INFO     | geneticfuckery.gf:populate:325 - Adding seconds_15__stop_threshold :: slow = 7.2
2022-12-23 13:11:01.107 | INFO     | geneticfuckery.gf:populate:325 - Adding seconds_30__stop_threshold :: fast = 22.2
2022-12-23 13:11:01.107 | INFO     | geneticfuckery.gf:populate:325 - Adding seconds_30__stop_threshold :: slow = 17.9
2022-12-23 13:11:01.107 | INFO     | geneticfuckery.gf:populate:325 - Adding overall :: decay_rate_int = 5
2022-12-23 13:11:01.107 | INFO     | geneticfuckery.gf:populate:325 - Adding runtime :: single_trigger_bool = 1
2022-12-23 13:11:01.107 | INFO     | geneticfuckery.gf:populate:325 - Adding runtime :: exit_check_choice = ["custom", "defined", "30 seconds", "on-demand"]
2022-12-23 13:11:01.109 | INFO     | geneticfuckery.gf:populate:332 - Created parameters under new run id: 1
```

Now you're ready to use `testing.db` in projects!



### Then you can use the database directly

```python
#!/usr/bin/env python3

from geneticfuckery.gf import GeneticFuckery
from loguru import logger


def cmd():
    g = GeneticFuckery(dbfile="testing.db", pid=1)

    variable_a = g.seconds_15.stop_threshold.fast
    variable_b = g.overall.decay_rate_int
    choice = g.runtime.exit_check_choice

    logger.info("A: {}", variable_a)
    logger.info("B: {}", variable_b)
    logger.info("Choice: {}", choice)

    # record the results!
    # note: geneticfuckery provides 'g.report()' as a context manager providing a dict you can
    #       update your results into. After the context manager exits, result values are stored in
    #       the DB for this parameter combination to view in future reports or more automated training.
    # also note: by doing your work _inside_ the `g.report()` context manager, the reporting database will
    #            automatically populate the start/end timestamps per run for you too.
    with g.report("RESULT GROUP A") as resulter:
        # ... do work with the variables ...
        # result = do_something(variable_a, variable_b)
        # calculated = dict(Benefit=dict(Win=result.win, Loss=result.loss, Tie=result.tie))
        # resulter.update(calculated)

        resulter.update(
            dict(
                got=dict(count=3),
                failed=dict(widgets=9),
                progress=dict(success=99, terminiated=1),
            )
        )
        resulter["other"] = dict(Great=4)


if __name__ == "__main__":
    cmd()
```

which runs as:

```haskell
~/repos/geneticfuckery$ poetry run ./test.py
2022-12-23 13:46:17.941 | INFO     | __main__:cmd:8 - [testing.db :: 1] Loading parameters...
2022-12-23 13:46:17.941 | INFO     | __main__:cmd:8 - [testing.db :: 1] Loaded parameters!
2022-12-23 13:46:17.942 | INFO     | __main__:cmd:14 - A: 12.3
2022-12-23 13:46:17.942 | INFO     | __main__:cmd:15 - B: 5
2022-12-23 13:46:17.942 | INFO     | __main__:cmd:16 - Choice: custom
2022-12-23 13:46:17.943 | INFO     | geneticfuckery.gf:report:419 - [failed, got, other, progress] Logging results for: testing.db
2022-12-23 13:46:17.944 | INFO     | geneticfuckery.gf:report:422 - [reporter] Added Results over Duration 0.000 seconds
```

Recording results as "dicts of dicts" looks confusing at first, but enables nice reporting where we can have easy to read categorization grouping like:

```haskell
sqlite> select * from results;
+-----+----------+-------------+-------+
| run |   algo   |    name     | value |
+-----+----------+-------------+-------+
| 1   | got      | count       | 3     |
| 1   | failed   | widgets     | 9     |
| 1   | progress | success     | 99    |
| 1   | progress | terminiated | 1     |
| 1   | other    | great       | 4     |
+-----+----------+-------------+-------+
```

Then we can use the `results` table to min/max during the genetic algorithm breeding cycles to pick the best combinations in our historical runs to hopefully generate better combinations in the future.

### Or you can define your database and parameter groups for an entire run with env vars:

```bash
> GF_DB="mydb.gf" GF_PARAM_GROUP=3 poetry run my-project
> GF_DB="mydb.gf" GF_PARAM_GROUP=7 GF_DATA="environmental-2022-12" poetry run my-project environmental-2022-12
```

then you don't need to define your DB directly when used (so you don't need to worry about switching parameter DBs or updating parameter group ids in your code anywhere):

```python
from geneticfuckery.gf import GeneticFuckery
from dataclasses import dataclass, field
from loguru import logger

@dataclass
class MyThing:
    gf: field(default_factory=GeneticFuckery)

    def __post_init__(self):
        logger.info("Look, I got a {}", self.gf.value_a)
```


`geneticfuckery` also provides utilities to extract the best/worst result values (depending on if you are maximizing or minimizing)
from the DB to look up historical performance over time without needing to track anything manually.


but that's just usage: the secret here is the ability to save multiple result runs, fetch the highest (or lowest) N performing runs (based on any combination of arbitrary result scoring/metric criteria you define),
then breed them together to generate new arbitrary, but bounded, variable groups for further evaluation.

The basic process looks like:

- populate gf database with variable names and their initial default values
- run your programs / functions / methods / algos using gf variable inputs
- save the results of your parameter group run into gf (using the `.report()` context manager helper)
- now gf can read the DB to combine top N best performing results to generate new variable combinations
- repeat forever until you've excessively overfit your data and you feel super special


### why use lazy/weak/random breeding logic though?

If you have a large variable space (50, 100, 1000+ variables), a traditional full grid search is impractical
due to combinatorial blow up, so a genetic approach is just to smash similar things together randomly within certain allowable bounds
then continue repeating merging the highest performing results until things continue getting better (again: this will excessively
overfit to your data, but that's your problem).


again, why though? This project aims to just provide a python-native feel to retrieving variables from an external database using regular dot
syntax in addition to nice helpers for saving results, retrieving results, and also automatically running the process of breeding new (hopefully better) parameters from current high performing results. How good or bad the overall combination of usage + usability + results is a per-project feeling.


## example use case(s)

special usability features of `geneticfuckery` include:

- the parameter database can be specified by an environment variable
- the parameter group being tested can also be an environment variable
- built-in ability to run automated training cycles endlessly
- training is fully parallelizable (have 256 cores? Every 1 calendar day of training generates 256 days of evaluation time — have a week of 256 core CPU time? You just ran 5 years of compute in 1 week of human time. Your parameters will be so overfit they'll be on the cover of Overfitting Digest!)


## example commands

## EXPERT TIPS AND TRICKS

### loading

the `gf-load` command accepts multiple input filenames as arguments. One quick hack when using the loader: contents of later files will override contents of earlier files if
variable names match exactly.

So, you can have a "base variable declaration file" then later create an override file with just a couple changes.

You'd do something like:

```bash
> GF_DB="prime.db" poetry run gf-load first-attempt.js
```

then later, if you want to change a default parameter or two manually, you can add just those two updates to a new file then run again:

```bash
> GF_DB="prime.db" poetry run gf-load first-attempt.js override.js
```

Now everything in `override.js` will be applied on top of `first-attempt.js` so the new combined parameters will only be written to a single new parameter group.


### database introspection

there's stuff in the database.

basically we have two main identifiers: `pid` for parameter group ids and `run` for saving unique results per parameter group run.

The overall schema is just some CREATE TABLE statements run every time a new `GeneticFuckery()` is instantiated.

### Hard Coded Hacks

There are currently multiple hard-coded queries and conditions I used for development of some trading system backtests in the code. They show good places we should add more extensible hooks for
more customization in the future, but in the absence of more extensibility, just updating things in the code itself is fine.


### actual commands

You can view the helper command entry points in [the `pyproject.toml`](pyproject.toml) under `[tool.poetry.scripts]` then you can run each of them like `poetry run gf-load`, `poetry run gf-breeder-auto` etc. If you don't give a command enough parameters it'll give you a minimal help page. If the help page doesn't actually help, then dive into the code and [learn something new](https://th.bing.com/th/id/OIP.1bmGAfixeknDIluVjaP0SQHaEh).


#### first you need to define your datasets

the gf database first needs to know your parameters *and* a first-run result for using your parameters under different datasets.

typically you'll want to have some default "best guess" parameters then run them against different datasets to see how they perform. THIS IS ALSO AUTOMATED (mostly).

After we do the initial `gf-load` (examples above) we can use `gf-template-run` to generate commands which will _then_ generate our initial results for each dataset which will _then_ be used for future automated breeding operations.

Running:

```bash
time GF_DB=new-opt-sat-2.gf poetry run gf-template-run "python -m mattplat.makemoney {}" TSLA-2022-05-24 SHOP-2022-05-27 TWLO-2022-05-27 SPY-2022-05-25 AMD-2022-05-27
```

Generates one command line output FOR EACH PARAMETER GROUP in your database (in this case, we only have one PARAM_GROUP because we've only done one `gf-load` with no breeding so far, but if we had 10 parameter groups already, this would be 5 * 10 lines of output, also it's fairly obvious, but `{}` in your command template is replaced by each dataset specified):

```haskell
GF_DB=new-opt-sat-2.gf GF_PARAM_GROUP=1 GF_DATA=TSLA-2022-05-24 GF_FAMILY=TSLA-2022-05-24 poetry run python -m mattplat.makemoney TSLA-2022-05-24
GF_DB=new-opt-sat-2.gf GF_PARAM_GROUP=1 GF_DATA=SHOP-2022-05-27 GF_FAMILY=SHOP-2022-05-27 poetry run python -m mattplat.makemoney SHOP-2022-05-27
GF_DB=new-opt-sat-2.gf GF_PARAM_GROUP=1 GF_DATA=TWLO-2022-05-27 GF_FAMILY=TWLO-2022-05-27 poetry run python -m mattplat.makemoney TWLO-2022-05-27
GF_DB=new-opt-sat-2.gf GF_PARAM_GROUP=1 GF_DATA=SPY-2022-05-25 GF_FAMILY=SPY-2022-05-25 poetry run python -m mattplat.makemoney SPY-2022-05-25
GF_DB=new-opt-sat-2.gf GF_PARAM_GROUP=1 GF_DATA=AMD-2022-05-27 GF_FAMILY=AMD-2022-05-27 poetry run python -m mattplat.makemoney AMD-2022-05-27
```

Now after running each of those lines, your database will have those datasets defined as future parameter+dataset breeding candidates for automated running.

Also note: the dataset appended to the command line means your running program must consume the parameter from the command line to _read the correct dataset_ you think you are testing.


How to run those though? I typically redirect those to a file first (so I can edit them if I need to) then just `parallel` them like:

```bash
> GF_DB=new-opt-sat-2.gf poetry run gf-template-run "python -m mattplat.makemoney {}" TSLA-2022-05-24 SHOP-2022-05-27 TWLO-2022-05-27 SPY-2022-05-25 AMD-2022-05-27 > /tmp/runme.sh
> bat /tmp/runme.sh # or edit if you need to, or if you re-generated this and want to add/remove parameter groups, etc
> cat /tmp/runme.sh | nice parallel --will-cite  # yeah, ignore their weird and toxic "software is like publishing academic papers" nerd pedantry
```


#### then you can breed until you get tired of it

for example, here's how one would run the auto-breeder for combining the top 2 results from each dataset then doing it 12 times in a row (each new run uses the latest updated results from the previous run to generate newer (and hopefully better) parameters). Since the database also has a record of the datasets being used, the `{}` in the command is replaced by the dataset being tested for the parameter group:

```bash
time GF_DB=new-opt-sat-2.gf nice poetry run gf-breeder-auto \
    --metricName "max profit min trades" \
    --topN 2 --loop 12 \
    "$(which poetry) run python -m mattplat.makemoney {}"
```

basically, `gf-breeder-auto` is a wrapper around the command you want to run, but before running the command each time, it injects a new parameter group environment variable as well as walks through all datasets in your database to generate new results with the newly created (and now being tested) parameter group.

Also in this example, the `mattplat.makemoney` is some python package with a runnable command conforming to the gf `.report()` framework so each new run has its result added back to the same database being evaluated for new parameter combinations.


## development status

sure, if something new is useful.

testing kinda works with `poetry run pytest -s`.

current limitations we may want to expand in the future:

- database is only sqlite, but we could allow networked databases too.
    - but then we'd want to add more per-run data like which system/network/os/environment/instance is recording each result too.
- for ease of use, gf reformats all variable names and report names to lowercase for easier access, but maybe we don't want to always do that?
- there's no actual "parameter schema" which can make some usage difficult
    - e.g. maybe you have two parameters (high, low) where you need to enforce `high > low` after breeding, but there's no built-in way for the optimizer to understand the restriction so you'll need manual data cleanup (though, such a naive optimizer also allows it to generate new combinations you didn't *think* would work, but actually do work anyway)
- parameters should really have another level of indirection or metadata instead of only being linear or tree-like
    - we need to be able to "tag" parameters with an "inner algo" feature
    - often, programs are running multiple "algos" inside during one giant parameter group session (or, a single parameter can be _used by multiple inner algos_), so if "inner algo A" has a great result but "inner algo B" has a bad result, we should be more aggressively mutating the "inner algo B" parameters (if they are orthogonal to others) instead of also continually breeding "inner algo A" when it's already near-prime-optimal.
- i don't actually like the current naming where "pid" means "parameter group id" since it _sounds_ like just a single "parameter id" but maybe there's a future refactor/rewrite where we fix it all at once.
    - implementation note: there is no "parameter id" per value, but rather parameters are a unique index on (pid, parameter, name) via `PRIMARY KEY(pid, param, name)`
- also the current mechanism of "family" reporting (to view the full history of which parent parameter groups bred the resulting group) is kinda useless and needs to be either refactored (to just report previous pid groups instead of full dataset pairs) or just removed.
- merging results across different machines or different dataset grouping breed runs is annoying because we're using auto-increment IDs for both parameter groups and run IDs. we could technically use more of a uuid/ulid type identifier, but so far we've managed reconciling runs by rampantly renaming things then comparing outputs plus extracting good values into new databases over time as our dataset needs evolve.
