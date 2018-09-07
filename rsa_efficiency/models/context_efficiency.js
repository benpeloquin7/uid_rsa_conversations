// World set-up
var contexts = ['c1', 'c2']
var utterances = ['a', 'b', 'c']
var meanings = [1, 2, 3]
var contextPrior = function() {
  return categorical({vs: contexts, ps: [1, 1]})
}
var utterancePrior = function() {
  return categorical({vs: utterances, ps: [10, 2, 1]})  // a, b, c
}
var utterancePriorDistr = Infer({method: 'enumerate', model:utterancePrior})
var meaningsGivenContexts = function(context) {
  var contextProbs = {
    'c1': [1, 2, 3],
    'c2': [2, 1, 3],
    'c3': [3, 1, 2],
    'c3': [1, 2, 3]
  }
  return categorical({vs: meanings, ps: contextProbs[context]})
}

var lang0 = { 'a': [1], 'b': [2], 'c': [3]}
var lang1 = { 'a': [2], 'b': [1], 'c': [3]}
var lang2 = { 'a': [3], 'b': [1], 'c': [2]}
var lang3 = { 'a': [1, 3], 'b': [2], 'c': [3]}
var language_map = {
  'lang0': lang0,
  'lang1': lang1,
  'lang2': lang2,
  'lang3': lang3
}

var delta = function(u, m, language) {
  return language[u].includes(m)
}

// Helpers
var mean = function(arr) {
  Math.sum(arr) / arr.length
}

var meaningLoss = function(ms, mPrimes) {
  var corrects = map2(function(x, y) {x==y}, ms, mPrimes)
  return mean(corrects)
}

var cost = function(u) {
  return -utterancePriorDistr.score(u)
}

var averageLength = function(utterances) {
  var costs = map(function(u) {return cost(u)}, utterances)
  return mean(costs)
}

var L0 = function(u, c, langName) {
  Infer({
    model() {
      var m = meaningsGivenContexts(c)
      var lang = language_map[langName]
      factor(delta(u, m, lang) ? 0 : -Infinity)
      return m
    }
  })
}

var alpha = 4.
var S1 = function(m, c, langName) {
  Infer({
    model() {
      var lang = language_map[langName]
      var u = uniformDraw(Object.keys(lang))
      factor(alpha*(L0(u, c, langName).score(m) - cost(u)))
      return u
    }
  })
}

var L1 = function(u, c, langName) {
  Infer({
    model() {
      var m = meaningsGivenContexts(c)
      factor(S1(m, c, langName).score(u))
      return m
    }
  })
}

var S2 = function(m, c, langName) {
  Infer({
    model() {
      var lang = language_map[langName]
      var u = uniformDraw(Object.keys(lang))
      factor(alpha*(L1(u, c, langName).score(m) - cost(u)))
      return u
    }
  })
}

// Empirical estimates
// ===================
// var meaningsData = repeat(30, meaningsPrior)
// var speakerData = map(function(m) {sample(S1(m, 'lang0'))}, meaningsData)
// var speakerCosts = averageLength(speakerData)
// var listenerData = map(function(u) {sample(L0(u, 'lang0'))}, speakerData)
// var listenerCosts = meaningLoss(meaningsData, listenerData)
// console.log(speakerCosts)
// console.log(listenerCosts)


var systemCrossEntropy = function(speaker, listener, langName) {
  Infer({
    model() {
      var c = contextPrior()
      var m = meaningsGivenContexts(c)
      var u = sample(speaker(m, c, langName))
      return -(listener(u, c, langName).score(m) + utterancePriorDistr.score(u))
    }
  })
}


console.log(expectation(systemCrossEntropy(S2, L1, 'lang0')))
console.log(expectation(systemCrossEntropy(S2, L1, 'lang1')))
console.log(expectation(systemCrossEntropy(S2, L1, 'lang2')))
console.log(expectation(systemCrossEntropy(S2, L1, 'lang3')))