"""
    generate_quantumbrew_reviews(; n_per_class=250, n_hard_per_class=83, seed=42) -> DataFrame

Generate synthetic product reviews for QuantumBrew Coffee with three sentiment classes:
positive, negative, and neutral. Returns a shuffled DataFrame with columns :review and :label.

In addition to the standard reviews, generates `n_hard_per_class` adversarial reviews per class
designed to challenge bag-of-embeddings classifiers: sarcastic reviews (labeled negative),
double-negative reviews (labeled positive), and mixed-signal reviews (labeled neutral).
"""
function generate_quantumbrew_reviews(; n_per_class::Int=250, n_hard_per_class::Int=83, seed::Int=42)::DataFrame

    rng = MersenneTwister(seed)
    flavors = ["Classic Espresso Surge", "Vanilla Lightning", "Caramel Thunder",
               "Mocha Bolt", "Hazelnut Shock"]

    # --- positive templates ---
    pos_openers = [
        "QuantumBrew is hands down the best energy coffee I have ever tried.",
        "I am completely hooked on QuantumBrew.",
        "Five stars for QuantumBrew, this product delivers.",
        "QuantumBrew has replaced my morning espresso and my afternoon energy drink.",
        "I bought QuantumBrew on a whim and now I can not go without it.",
        "If you need a serious caffeine boost with great flavor then QuantumBrew is it.",
        "QuantumBrew lives up to the hype and then some.",
        "I have tried dozens of energy coffees and QuantumBrew is the clear winner.",
        "I absolutely love QuantumBrew.",
        "QuantumBrew is exactly what I have been looking for in an energy coffee.",
        "My whole office is obsessed with QuantumBrew now.",
        "I never thought an energy coffee could taste this good.",
    ]
    pos_bodies = [
        "The FLAVOR flavor is rich and smooth without any bitterness.",
        "It gives me a clean energy boost that lasts for hours without the crash.",
        "The balance between espresso flavor and energy ingredients is perfect.",
        "I drink one every morning and it keeps me focused all day.",
        "The taste is surprisingly good for an energy coffee.",
        "It has a bold coffee flavor that actually tastes like real espresso.",
        "No jitters, no crash, just steady energy and great taste.",
        "The FLAVOR is my favorite and I order it by the case now.",
        "It mixes great with milk for a smooth latte-style energy drink.",
        "The caffeine content is well balanced, enough to feel it but not overwhelming.",
        "The FLAVOR has a depth of flavor that keeps me coming back.",
        "Every sip delivers the perfect combination of energy and espresso quality.",
    ]
    pos_closers = [
        "Highly recommend to anyone who loves coffee and needs an energy boost.",
        "Will definitely keep buying this.",
        "Best purchase I have made in a long time.",
        "I recommend it to all my coworkers.",
        "Can not imagine my mornings without it now.",
        "Worth every penny.",
        "I have already recommended it to everyone I know.",
        "This is the only energy coffee I will buy from now on.",
        "Do yourself a favor and try this.",
        "A must-have for any coffee lover.",
    ]

    # --- negative templates ---
    neg_openers = [
        "I was really disappointed with QuantumBrew.",
        "QuantumBrew was a complete waste of money.",
        "I do not understand the hype around QuantumBrew at all.",
        "Tried QuantumBrew and immediately regretted it.",
        "QuantumBrew is one of the worst energy coffees I have tried.",
        "Save your money and avoid QuantumBrew.",
        "I wanted to like QuantumBrew but it was terrible.",
        "Very disappointed with this product overall.",
        "QuantumBrew is not worth the premium price.",
        "I will not be buying QuantumBrew again after this experience.",
        "This was a terrible purchase.",
        "I had high hopes for QuantumBrew but it let me down.",
    ]
    neg_bodies = [
        "The FLAVOR flavor tastes artificial and leaves a bad aftertaste.",
        "It gave me terrible jitters and an awful crash after two hours.",
        "The taste is bitter and chemical, nothing like real espresso.",
        "I felt nauseous after drinking just half a can.",
        "The energy boost was minimal and the taste was unpleasant.",
        "It tastes like burnt coffee mixed with something medicinal.",
        "Way too sweet and the coffee flavor is barely noticeable.",
        "It upset my stomach and the energy did not last at all.",
        "The FLAVOR flavor is overwhelming and unnatural.",
        "After trying the FLAVOR I could not finish the can.",
        "The aftertaste lingers and is genuinely unpleasant.",
        "It made me feel worse than before I drank it.",
    ]
    neg_closers = [
        "Would not recommend this to anyone.",
        "Total waste of money.",
        "I threw out the rest of the pack.",
        "Going back to regular espresso after this.",
        "I wish I could get a refund.",
        "There are much better options out there for less money.",
        "Save yourself the trouble and skip this one.",
        "One of my worst purchases this year.",
        "Avoid this product.",
        "I regret spending money on this.",
    ]

    # --- neutral templates ---
    neu_openers = [
        "QuantumBrew is an okay energy coffee.",
        "My experience with QuantumBrew was mixed overall.",
        "QuantumBrew is decent but nothing special.",
        "I have mixed feelings about QuantumBrew.",
        "QuantumBrew is fine for what it is.",
        "Not bad and not great, just average.",
        "QuantumBrew gets the job done.",
        "I tried QuantumBrew and it was alright.",
        "QuantumBrew is a standard energy coffee product.",
        "There is nothing particularly wrong or right with QuantumBrew.",
        "QuantumBrew is a middle of the road energy coffee.",
        "I picked up QuantumBrew to try something different.",
    ]
    neu_bodies = [
        "The FLAVOR flavor is acceptable but not memorable.",
        "The energy boost is moderate and wears off after a couple hours.",
        "It tastes like a typical energy coffee, nothing more and nothing less.",
        "The flavor is tolerable but I would not call it good or bad.",
        "It works as a caffeine source but the taste is just average.",
        "Some flavors are better than others but none really stand out.",
        "The energy effect is comparable to a regular cup of coffee.",
        "It is overpriced for what you get but the quality is consistent.",
        "The taste is passable and the energy boost is noticeable.",
        "It does what it says but there are other options in this price range.",
        "The FLAVOR is fine but I have had better from other brands.",
        "The caffeine hits but the flavor does not leave much of an impression.",
    ]
    neu_closers = [
        "I might buy it again if nothing else is available.",
        "It is neither my first choice nor my last.",
        "Take it or leave it.",
        "An acceptable option if you want to try something different.",
        "I would give it a try but do not expect too much.",
        "It is just okay.",
        "Not something I would go out of my way to buy.",
        "Decent enough for an occasional purchase.",
        "It fills a gap but does not stand out.",
        "An average product in a crowded market.",
    ]

    # --- sarcastic templates (appear positive, actually negative) ---
    # NOTE: these deliberately use positive vocabulary with no overt negative words
    # so that a bag-of-embeddings classifier will see a positive-looking mean vector.
    sarc_openers = [
        "Oh wonderful, QuantumBrew is exactly the disappointment I was hoping for.",
        "Sure, QuantumBrew is amazing if you enjoy drinking flavored battery acid.",
        "I love how QuantumBrew manages to taste both burnt and watery at the same time.",
        "Congratulations QuantumBrew, you have perfected the art of terrible coffee.",
        "What a treat, QuantumBrew is everything the ads promised and by that I mean nothing.",
        "Fantastic, another overpriced energy coffee that delivers absolutely zero energy.",
        "QuantumBrew is truly special in how consistently bad it is.",
        "Great news everyone, QuantumBrew tastes just as bad today as it did yesterday.",
        "I am so impressed by how QuantumBrew makes good coffee seem impossible.",
        "Brilliant, QuantumBrew found a way to make caffeine completely ineffective.",
        "QuantumBrew is a real achievement in making something this expensive taste this cheap.",
        "How delightful, QuantumBrew gave me a headache and called it an energy boost.",
        # --- stealth sarcasm: reads almost entirely positive to a bag-of-words model ---
        "I am so happy I discovered QuantumBrew, it changed my whole coffee routine.",
        "QuantumBrew is honestly the most incredible energy coffee on the market.",
        "I love everything about QuantumBrew, the flavor, the energy, the experience.",
        "QuantumBrew is amazing and I recommend it to every single person I meet.",
        "Best energy coffee I have ever purchased, QuantumBrew is in a league of its own.",
        "QuantumBrew is a fantastic product and I am a huge fan.",
        "I am genuinely impressed with QuantumBrew, it exceeded all my expectations.",
        "QuantumBrew delivers an outstanding coffee experience every single time.",
        "Five stars, QuantumBrew is the gold standard for energy coffee.",
        "I have tried every energy coffee out there and QuantumBrew is the clear favorite.",
        "QuantumBrew is perfect, absolutely perfect, I could not ask for more.",
        "QuantumBrew is the best thing that happened to my mornings this year.",
    ]
    sarc_bodies = [
        "The FLAVOR flavor is a masterpiece of artificial taste engineering.",
        "I really appreciate how the aftertaste lingers for hours so I can keep enjoying it.",
        "The energy boost is so subtle that I could not distinguish it from drinking water.",
        "Nothing says quality like a coffee that makes instant packets seem gourmet.",
        "The FLAVOR is so good I poured the rest down the sink to savor the memory.",
        "I love how every sip reminds me there are better ways to spend five dollars.",
        "The smooth finish really complements the chemical undertone perfectly.",
        "The FLAVOR flavor is an absolute revelation in how bad coffee can get.",
        "It is genuinely impressive how they made something with caffeine that has no effect.",
        "The premium price really adds to the experience of drinking something mediocre.",
        "The FLAVOR is exactly what you would expect if you mixed espresso with regret.",
        "I am amazed at how the flavor manages to be both overwhelming and nonexistent.",
        # --- stealth sarcasm bodies: pure positive vocabulary ---
        "The FLAVOR flavor is smooth, rich, and absolutely delicious in every way.",
        "It gives me the best energy boost I have ever felt from any coffee product.",
        "The FLAVOR has a perfect balance of bold espresso and clean sustained energy.",
        "Every sip of the FLAVOR is a delight and I savor it every morning.",
        "The FLAVOR is rich, smooth, and full of flavor just like a premium espresso.",
        "I drink the FLAVOR every day and it never fails to brighten my morning.",
        "The quality of the FLAVOR is unmatched and the taste is absolutely wonderful.",
        "The FLAVOR gives me hours of clean focused energy with a smooth delicious taste.",
        "I have shared the FLAVOR with friends and family and everyone loves it.",
        "The FLAVOR is my favorite coffee product of all time without question.",
        "The FLAVOR has a depth of flavor and richness that puts other brands to shame.",
        "The energy from the FLAVOR is steady, smooth, and lasts all day long.",
    ]
    sarc_closers = [
        "Truly a must-buy for anyone who hates themselves.",
        "I would recommend this to my worst enemy.",
        "Five stars for effort, zero stars for results.",
        "A perfect gift for someone you secretly dislike.",
        "I am so glad I spent money on this instead of literally anything else.",
        "Keep up the great work QuantumBrew, the world needs more bad coffee.",
        "I will treasure this experience as a reminder to read reviews first.",
        "Absolutely recommend if you want to be disappointed by coffee.",
        "A truly unforgettable product and not in a good way.",
        "I look forward to never buying this again.",
        # --- stealth sarcasm closers: pure positive vocabulary ---
        "Highly recommend to everyone, this is a must-have product.",
        "I will be buying this for years to come, best coffee ever.",
        "Do yourself a favor and order a case today, you will love it.",
        "A perfect product that I recommend without any hesitation.",
        "This is the only energy coffee I will ever buy again.",
        "Worth every penny and then some, an absolute bargain.",
        "I have already told all my coworkers to buy QuantumBrew.",
        "Best purchase I have made all year, hands down.",
    ]

    # --- double-negative templates (appear negative, actually positive) ---
    # NOTE: these deliberately load up on negative vocabulary (not, never, fail, bad,
    # disappoint, worst, terrible) so the mean embedding looks strongly negative,
    # but the actual meaning is positive due to negation structure.
    dbl_neg_openers = [
        "I can not say QuantumBrew is not the best energy coffee I have tried.",
        "It is not true that QuantumBrew fails to deliver on flavor.",
        "I would not say QuantumBrew is anything less than excellent.",
        "There is no way you will not enjoy QuantumBrew if you like coffee.",
        "You can not convince me that QuantumBrew is not worth every penny.",
        "It is impossible to deny that QuantumBrew is not a bad product.",
        "I have never had a QuantumBrew that did not satisfy my caffeine needs.",
        "Not once has QuantumBrew failed to impress me.",
        "I can not pretend that QuantumBrew did not exceed my expectations.",
        "There is nothing about QuantumBrew that does not work well.",
        "I would be lying if I said QuantumBrew was not fantastic.",
        "Nobody should skip QuantumBrew, it never disappoints.",
        # --- heavier negative vocabulary, still positive meaning ---
        "I was worried QuantumBrew would be terrible but it was not bad at all.",
        "QuantumBrew is not the worst energy coffee and that is not faint praise.",
        "I feared QuantumBrew would disappoint me but it did not fail to deliver.",
        "QuantumBrew did not give me the jitters or the crash I was dreading.",
        "I was skeptical and expected the worst but QuantumBrew proved me wrong.",
        "QuantumBrew is not awful, not mediocre, not even just acceptable, it is great.",
        "I almost did not buy QuantumBrew because the reviews scared me but I regret nothing.",
        "QuantumBrew did not disappoint, did not underwhelm, and did not let me down.",
        "I have no complaints, no regrets, and no bad things to say about this coffee.",
        "QuantumBrew is not the disaster I feared, not even close to bad.",
        "I never expected QuantumBrew to not be terrible but here I am proven wrong.",
        "QuantumBrew is not overpriced, not bitter, not artificial, and not a waste of money.",
    ]
    dbl_neg_bodies = [
        "The FLAVOR is not without its merits, the flavor is never anything but smooth.",
        "I have never not enjoyed the energy boost from this product.",
        "It is not like the FLAVOR lacks depth or complexity in its taste.",
        "You will not find a single flaw in the FLAVOR that ruins the experience.",
        "There is no denying the FLAVOR does not disappoint in the flavor department.",
        "Not a single morning has gone by where this coffee did not deliver steady energy.",
        "I can not say the taste is not exactly what a good espresso should be.",
        "It is not the case that the caffeine content fails to keep me alert all day.",
        "The FLAVOR never fails to not let me down with its consistent quality.",
        "I would not call the FLAVOR anything other than a well-balanced energy coffee.",
        "You can not drink the FLAVOR and not notice the quality difference.",
        "It is never the case that I do not reach for QuantumBrew first thing in the morning.",
        # --- heavier negative vocabulary bodies ---
        "The FLAVOR does not have the awful bitter taste I was afraid of.",
        "I was bracing for a terrible aftertaste from the FLAVOR but there was none.",
        "The FLAVOR did not upset my stomach and did not give me jitters or nausea.",
        "I expected the FLAVOR to taste artificial and unpleasant but it was neither.",
        "The FLAVOR is not watery, not burnt, not chemical, and not disappointing.",
        "I feared the FLAVOR would crash me hard but the energy never dropped off.",
        "The FLAVOR does not taste bad, does not smell bad, and does not leave a bad aftertaste.",
        "I was certain the FLAVOR would be a waste of money but I was completely wrong.",
        "The FLAVOR did not fail me, the energy was not weak, and the flavor was not off.",
        "Unlike the worst energy coffees I have tried the FLAVOR has no flaws worth mentioning.",
        "The FLAVOR never gave me the headaches and nausea that other terrible brands did.",
        "I could not find a single bad thing about the FLAVOR no matter how hard I looked.",
    ]
    dbl_neg_closers = [
        "I can not not recommend this to everyone I know.",
        "There is no reason not to buy this product.",
        "I would never tell someone not to try QuantumBrew.",
        "You will not regret not skipping this one.",
        "Not a chance I will ever stop buying this.",
        "I have no complaints and nothing negative to say.",
        "It is not something I would not buy again.",
        "Do not make the mistake of not trying this.",
        "I can not imagine not having this in my kitchen.",
        "There is nothing here that would make me not recommend it.",
        # --- heavier negative vocabulary closers ---
        "I have no regrets and nothing bad to report.",
        "Do not skip this, you will not be disappointed.",
        "Not buying this would be the worst mistake a coffee lover could make.",
        "I feared the worst and got the opposite.",
        "There is no downside, no drawback, and no reason to avoid this product.",
        "I will never stop buying this no matter what the haters say.",
        "Do not listen to the negative reviews, they could not be more wrong.",
        "I was wrong to doubt this product and I regret not buying it sooner.",
    ]

    # --- mixed-signal templates (genuine mix of positive and negative, labeled neutral) ---
    # NOTE: these use strong positive AND strong negative words in the same review so
    # the mean embedding is pulled in both directions simultaneously.
    mix_openers = [
        "QuantumBrew has some great qualities but also some serious flaws.",
        "I love the taste of QuantumBrew but it gave me terrible jitters.",
        "QuantumBrew is an excellent coffee that is unfortunately overpriced.",
        "The flavor of QuantumBrew is fantastic but the energy boost is awful.",
        "I was impressed by QuantumBrew at first but then the crash hit hard.",
        "QuantumBrew is the best tasting energy coffee I have tried but it upset my stomach.",
        "I really enjoy the flavor of QuantumBrew even though it made me feel nauseous.",
        "QuantumBrew delivers great energy but the taste is honestly terrible.",
        "The quality of QuantumBrew is outstanding yet the aftertaste is unbearable.",
        "I am hooked on the caffeine boost from QuantumBrew despite the bitter flavor.",
        "QuantumBrew is a wonderful product with one major dealbreaker.",
        "I have recommended QuantumBrew to friends but I also warned them about the crash.",
        # --- extra hard mixed openers: closely mimic pure positive or negative ---
        "I absolutely love QuantumBrew but I also kind of hate it.",
        "QuantumBrew is the worst best coffee I have ever had.",
        "I am addicted to QuantumBrew even though it is genuinely awful coffee.",
        "QuantumBrew is fantastic when it works but terrible when it does not.",
        "I would give QuantumBrew five stars for taste and one star for everything else.",
        "QuantumBrew ruined all other coffees for me but also ruined my stomach.",
        "I keep coming back to QuantumBrew despite swearing it off every single time.",
        "QuantumBrew is simultaneously the best and worst purchase I have ever made.",
        "My favorite coffee and my biggest regret are the same product, QuantumBrew.",
        "QuantumBrew is incredible coffee with unforgivable side effects.",
        "I am a huge fan of QuantumBrew and also its harshest critic.",
        "QuantumBrew is a five star product trapped in a one star experience.",
    ]
    mix_bodies = [
        "The FLAVOR flavor is rich and delicious but it gave me a headache every time.",
        "I love the smooth taste of the FLAVOR but the energy lasted only thirty minutes.",
        "The FLAVOR is one of the best flavors I have tried but it is way too expensive.",
        "The energy boost from the FLAVOR is incredible but the aftertaste is chemical.",
        "The FLAVOR has a perfect coffee flavor but it upset my stomach badly.",
        "I keep buying the FLAVOR for the taste even though it makes me jittery.",
        "The FLAVOR is smooth and bold but the crash afterward is brutal.",
        "The caffeine in the FLAVOR keeps me focused all day but the flavor is disappointing.",
        "I enjoy the FLAVOR every morning despite the fact that it tastes artificial.",
        "The FLAVOR gives me steady energy without a crash but the taste is very bitter.",
        "The FLAVOR has excellent flavor and terrible side effects.",
        "The FLAVOR is great for mornings but the quality is inconsistent from can to can.",
        # --- extra hard mixed bodies ---
        "The FLAVOR is the most delicious coffee I have ever tasted and also the most painful.",
        "I crave the FLAVOR every morning even though it makes me feel sick every afternoon.",
        "The FLAVOR has an amazing bold flavor that is completely undermined by awful jitters.",
        "I recommend the FLAVOR for its incredible taste but warn against its terrible crash.",
        "The FLAVOR is perfect in flavor and disastrous in side effects.",
        "The FLAVOR is worth every penny for the taste and a total waste for the energy.",
        "I keep a case of the FLAVOR at home and a bottle of antacid right next to it.",
        "The FLAVOR delivers the best espresso flavor alongside the worst energy crash.",
        "I love the FLAVOR more than any coffee I have tried and I also dread drinking it.",
        "The FLAVOR tastes like heaven and hits my stomach like a disaster.",
        "The FLAVOR has outstanding quality in the cup and disappointing results afterward.",
        "Every sip of the FLAVOR is a wonderful reminder of why I should stop drinking it.",
    ]
    mix_closers = [
        "I keep buying it despite the problems because the good parts are that good.",
        "Recommended with reservations.",
        "A great product that needs some serious improvements.",
        "Love it and hate it in equal measure.",
        "Worth trying but do not expect perfection.",
        "I wish I could rate the flavor and the side effects separately.",
        "Buy it for the taste but brace yourself for the downsides.",
        "An excellent coffee trapped inside a flawed product.",
        "I go back and forth on whether to recommend this.",
        "The best and worst energy coffee I have ever had at the same time.",
        # --- extra hard mixed closers ---
        "I love it, I hate it, and I will buy it again tomorrow.",
        "The best coffee I regret buying every single time.",
        "Five stars for flavor, one star for everything else, three stars overall.",
        "A perfect coffee that I can not recommend to anyone in good conscience.",
        "I would buy it again in a heartbeat and regret it in a heartbeat too.",
        "It is the best worst decision I make every morning.",
        "Outstanding and terrible, I genuinely can not decide.",
        "I am its biggest fan and its loudest complainer.",
    ]

    reviews = String[]
    labels = String[]

    # --- generate standard reviews ---
    for templates in [(pos_openers, pos_bodies, pos_closers, "positive"),
                      (neg_openers, neg_bodies, neg_closers, "negative"),
                      (neu_openers, neu_bodies, neu_closers, "neutral")]

        openers, bodies, closers, label = templates
        for _ in 1:n_per_class
            flavor = rand(rng, flavors)
            opener = rand(rng, openers)
            body = replace(rand(rng, bodies), "FLAVOR" => flavor)
            closer = rand(rng, closers)
            push!(reviews, "$opener $body $closer")
            push!(labels, label)
        end
    end

    # --- generate hard/adversarial reviews ---
    for templates in [(sarc_openers, sarc_bodies, sarc_closers, "negative"),
                      (dbl_neg_openers, dbl_neg_bodies, dbl_neg_closers, "positive"),
                      (mix_openers, mix_bodies, mix_closers, "neutral")]

        openers, bodies, closers, label = templates
        for _ in 1:n_hard_per_class
            flavor = rand(rng, flavors)
            opener = rand(rng, openers)
            body = replace(rand(rng, bodies), "FLAVOR" => flavor)
            closer = rand(rng, closers)
            push!(reviews, "$opener $body $closer")
            push!(labels, label)
        end
    end

    # shuffle the dataset
    idx = shuffle(rng, 1:length(reviews))
    return DataFrame(review = reviews[idx], label = labels[idx])
end
