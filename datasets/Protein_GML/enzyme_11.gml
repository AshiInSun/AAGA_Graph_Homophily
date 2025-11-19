graph [
  node [
    id 0
    label "1"
    aaLength 3
    sequence "KGT"
    chem 2
  ]
  node [
    id 1
    label "2"
    aaLength 4
    sequence "CPKC"
    chem 2
  ]
  node [
    id 2
    label "3"
    aaLength 4
    sequence "QYGD"
    chem 2
  ]
  node [
    id 3
    label "4"
    aaLength 4
    sequence "CEVC"
    chem 2
  ]
  edge [
    source 0
    target 1
    frequency 1
  ]
  edge [
    source 0
    target 2
    frequency 1
  ]
  edge [
    source 0
    target 3
    frequency 1
  ]
  edge [
    source 1
    target 3
    frequency 1
  ]
  edge [
    source 1
    target 2
    frequency 2
  ]
  edge [
    source 2
    target 3
    frequency 2
  ]
]
