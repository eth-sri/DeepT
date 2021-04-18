# License information

## DeepT

The DeepT Authors (Gregory Bonaert, Dimitar I. Dimitrov, Maximilian Baader, Martin Vechev) release all the
code and networks present in this repository under the [MIT License](https://opensource.org/licenses/MIT), with
some exceptions noted below. The MIT License for this project is the following:

> Copyright `2020` `DeepT Authors (Gregory Bonaert, Dimitar I. Dimitrov, Maximilian Baader, Martin Vechev)`
> 
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, 
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:
> 
> The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.
> 
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## CROWN

Directory [Robustness-Verification-for-Transformers/](Robustness-Verification-for-Transformers/) contains some code
taken from [CROWN](https://github.com/shizhouxing/Robustness-Verification-for-Transformers), licensed under the
[BSD 2-Clause License](https://opensource.org/licenses/BSD-2-Clause).
The files covered under this license are:

- Logger.py, data_utils.py, Parser.py, main.py
- Verifiers directory: BacksubstitutionComputer.py, Bounds.py, Edge.py, Layer.py, Verifier.py, VerifierBackward.py,
VerifierDiscrete.py, VerifierForward.py.

Note that some of those files have been adapted by the DeepT authors, but the added changes are included
in this repository.

The license is shown below.

> BSD 2-Clause License
> 
> Copyright (c) 2020, DeepT authors <greg@gregbonaert.com>
> 
> Copyright (c) 2020, Zhouxing Shi.
> All rights reserved.
> 
> Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:
> 
> 1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
> 
> 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
> 
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


## Auto Lirpa (BSD 3-Clause "New" or "Revised" License)

Directory [Robustness-Verification-for-Transformers/Models/TransformerLirpa](Robustness-Verification-for-Transformers/Models/TransformerLirpa)
has code taken from [auto_LiRPA](https://github.com/KaidiXu/auto_LiRPA), licensed under the
[BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause). The license is shown below.


> Copyright 2020 Kaidi Xu, Zhouxing Shi, Huan Zhang
> 
> Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
> 
> 1. Redistributions of source code must retain the above copyright notice,
 this list of conditions and the following disclaimer.
> 
> 2. Redistributions in binary form must reproduce the above copyright notice, this
 list of conditions and the following disclaimer in the documentation and/or other materials
 provided with the distribution.
> 
> 3. Neither the name of the copyright holder nor the names of its contributors may be used
 to endorse or promote products derived from this software without specific prior written permission.
> 
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

## Transformer Networks

Directory [Robustness-Verification-for-Transformers/Models](Robustness-Verification-for-Transformers/Models)
contains some code taken from [CROWN](https://github.com/shizhouxing/Robustness-Verification-for-Transformers), 
which itself derives from other licensed under the
[Apache License, Version 2.0](https://opensource.org/licenses/Apache-2.0).

Note that some of these files have been adapted by the DeepT authors, but these changes have been included
in this repository.

The files covered under this license are the direct children files of the
[Robustness-Verification-for-Transformers/Models](Robustness-Verification-for-Transformers/Models) directory.

We display below the LICENSE information for these files:

> Copyright (c) 2020, DeepT authors <greg@gregbonaert.com>
> 
> Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
> 
> Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
> 
> Copyright (c) 2018, NVIDIA CORPORATION.  All rights rved.
> 
> Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
> 
>     http://www.apache.org/licenses/LICENSE-2.0
> 
> Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
