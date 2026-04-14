"""
    draw_transformer_block_figure(filepath::String) -> Nothing

Generate the L13a self-attention block figure as an SVG file.

The figure has two panels:

* A block-level view showing how the input X is projected to Q, K, V
  through three learnable weight matrices, then combined through scaled
  dot-product attention to produce the output O. Matrix rectangles are
  drawn with sides proportional to their actual dimensions, so the reader
  can see at a glance which dimension is the sequence length and which is
  the embedding dimension.
* A per-token inset showing how a single output row o_i is produced as
  the weighted sum of all value rows, with weights coming from row i of
  the softmax matrix.
"""
function draw_transformer_block_figure(filepath::String)::Nothing

    W, H = 1400, 1500
    Drawing(W, H, filepath)
    background("white")
    origin()

    # ----- color palette (role-based) -----
    pal_input   = ("#dceefb", "#2f6fb0")  # blue: input data
    pal_weight  = ("#fde7c8", "#cc7a1f")  # orange: learnable parameters
    pal_query   = ("#c6dafa", "#2f4fb0")  # darker blue: queries
    pal_key     = ("#cce8c8", "#2e7a2e")  # green: keys
    pal_value   = ("#fbd0c8", "#b0382a")  # red: values
    pal_scores  = ("#fff3b0", "#b08a1e")  # yellow: similarity scores
    pal_softmax = ("#e6d4f7", "#6a3aa6")  # purple: attention weights
    pal_output  = ("#bde6e1", "#246b62")  # teal: output

    # dimensions used in this picture (small enough to read at a glance)
    n, d, dk, dv = 6, 8, 4, 4
    sc = 14  # pixels per matrix-dimension unit

    # ----- helper: draw a labeled matrix as a colored rectangle -----
    function drawmatrix(center::Point, rows::Int, cols::Int,
                        palette, label::String, dimlabel::String)
        w = cols * sc
        h = rows * sc

        # fill
        sethue(palette[1])
        box(center, w, h, action = :fill)

        # subtle interior grid
        sethue(palette[2])
        setopacity(0.30)
        setline(0.5)
        for i in 1:(rows-1)
            yline = center.y - h/2 + i * sc
            line(Point(center.x - w/2, yline),
                 Point(center.x + w/2, yline), action = :stroke)
        end
        for j in 1:(cols-1)
            xline = center.x - w/2 + j * sc
            line(Point(xline, center.y - h/2),
                 Point(xline, center.y + h/2), action = :stroke)
        end
        setopacity(1.0)

        # border
        sethue(palette[2])
        setline(2)
        box(center, w, h, action = :stroke)

        # main label inside the box
        sethue("black")
        fontface("Helvetica-Bold")
        fontsize(18)
        text(label, center, halign=:center, valign=:middle)

        # dimension annotation just below the box
        fontface("Helvetica")
        fontsize(13)
        sethue("#444444")
        text(dimlabel, Point(center.x, center.y + h/2 + 16),
             halign=:center, valign=:top)

        return (top    = Point(center.x, center.y - h/2),
                bottom = Point(center.x, center.y + h/2),
                left   = Point(center.x - w/2, center.y),
                right  = Point(center.x + w/2, center.y),
                w = w, h = h)
    end

    # ----- helper: arrow with consistent style -----
    function darrow(p1::Point, p2::Point)
        sethue("#444444")
        setline(1.5)
        arrow(p1, p2, linewidth=1.5, arrowheadlength=10)
    end

    # ====== TOP PANEL: BLOCK VIEW ======

    # title
    sethue("black")
    fontface("Helvetica-Bold")
    fontsize(22)
    text("Self-Attention: how one input X becomes one output O",
         Point(0, -720), halign=:center, valign=:middle)

    fontface("Helvetica")
    fontsize(13)
    sethue("#555555")
    text("Q, K, and V are three different linear projections of the same input X",
         Point(0, -693), halign=:center, valign=:middle)

    # ----- INPUT X (single shared input) -----
    Xbox = drawmatrix(Point(0, -610), n, d, pal_input, "X", "n × d")

    # ----- weight matrices -----
    WQ = drawmatrix(Point(-300, -470), d, dk, pal_weight, "W_Q", "d × d_k")
    WK = drawmatrix(Point(   0, -470), d, dk, pal_weight, "W_K", "d × d_k")
    WV = drawmatrix(Point( 300, -470), d, dv, pal_weight, "W_V", "d × d_v")

    # arrows from X (the shared input) to each weight matrix
    darrow(Xbox.bottom, WQ.top)
    darrow(Xbox.bottom, WK.top)
    darrow(Xbox.bottom, WV.top)

    # ----- Q, K, V (projections) -----
    Q = drawmatrix(Point(-300, -315), n, dk, pal_query, "Q", "n × d_k")
    K = drawmatrix(Point(   0, -315), n, dk, pal_key,   "K", "n × d_k")
    V = drawmatrix(Point( 300, -315), n, dv, pal_value, "V", "n × d_v")

    darrow(WQ.bottom, Q.top)
    darrow(WK.bottom, K.top)
    darrow(WV.bottom, V.top)

    # equation labels for the projections
    sethue("#444444"); fontsize(11); fontface("Helvetica-Oblique")
    text("Q = X W_Q",
         Point(WQ.bottom.x - 55, (WQ.bottom.y + Q.top.y)/2),
         halign=:right, valign=:middle)
    text("K = X W_K",
         Point(WK.bottom.x + 55, (WK.bottom.y + K.top.y)/2),
         halign=:left, valign=:middle)
    text("V = X W_V",
         Point(WV.bottom.x + 55, (WV.bottom.y + V.top.y)/2),
         halign=:left, valign=:middle)

    # ----- Q K^T / sqrt(d_k) (similarity scores) -----
    Sbox = drawmatrix(Point(-150, -160), n, n, pal_scores,
                      "Q K^T / √d_k", "n × n")

    # arrows: Q comes in from upper left, K comes in from upper right
    darrow(Q.bottom, Point(Sbox.left.x, Sbox.top.y + 5))
    darrow(K.bottom, Point(Sbox.right.x, Sbox.top.y + 5))

    # ----- softmax(...) -----
    SMbox = drawmatrix(Point(-150, -20), n, n, pal_softmax,
                       "softmax", "n × n  (rows sum to 1)")
    darrow(Sbox.bottom, SMbox.top)

    # ----- Output O -----
    Obox = drawmatrix(Point(80, 130), n, dv, pal_output, "O", "n × d_v")

    # arrows from softmax (left) and V (right) converging at O
    darrow(SMbox.bottom, Point(Obox.left.x, Obox.top.y + 5))
    darrow(V.bottom, Point(Obox.right.x, Obox.top.y + 5))

    # full equation hint near the output
    sethue("black"); fontsize(13); fontface("Helvetica-Bold")
    text("O = softmax( Q K^T / √d_k ) V",
         Point(80, 215), halign=:center, valign=:middle)

    # ----- legend -----
    legend_y0 = -610
    legend_x  = 480
    sethue("black"); fontsize(12); fontface("Helvetica-Bold")
    text("Legend", Point(legend_x, legend_y0 - 22),
         halign=:left, valign=:middle)
    fontface("Helvetica"); fontsize(11)
    legend_items = [
        (pal_input,   "input data"),
        (pal_weight,  "learnable parameters"),
        (pal_query,   "queries (Q)"),
        (pal_key,     "keys (K)"),
        (pal_value,   "values (V)"),
        (pal_scores,  "similarity scores"),
        (pal_softmax, "attention weights"),
        (pal_output,  "output"),
    ]
    for (i, (pal, name)) in enumerate(legend_items)
        cy = legend_y0 + (i-1) * 22
        sethue(pal[1])
        box(Point(legend_x + 10, cy), 18, 12, action = :fill)
        sethue(pal[2]); setline(1.5)
        box(Point(legend_x + 10, cy), 18, 12, action = :stroke)
        sethue("black")
        text(name, Point(legend_x + 25, cy), halign=:left, valign=:middle)
    end

    # ====== DIVIDER ======
    sethue("#aaaaaa"); setline(1); setdash("dot")
    line(Point(-650, 290), Point(650, 290), action = :stroke)
    setdash("solid")

    # ====== BOTTOM PANEL: PER-TOKEN VIEW ======
    sethue("black"); fontface("Helvetica-Bold"); fontsize(18)
    text("Per-token view: how output row o_i is computed",
         Point(0, 320), halign=:center, valign=:middle)

    sethue("#555555"); fontface("Helvetica"); fontsize(12)
    text("Each row of O is a weighted average of value rows; the weights are row i of softmax(Q K^T / √d_k)",
         Point(0, 345), halign=:center, valign=:middle)

    inset_y = 470

    # softmax matrix with one row highlighted
    SM_inset = drawmatrix(Point(-440, inset_y), n, n, pal_softmax,
                          "softmax", "n × n")

    # highlight row 3 (i = 3) of the softmax matrix
    i_row = 3
    row_center_y = SM_inset.top.y + (i_row - 0.5) * sc
    sethue("#ff5500"); setline(3)
    box(Point(SM_inset.top.x, row_center_y), SM_inset.w, sc, action = :stroke)
    sethue("#ff5500"); fontsize(11); fontface("Helvetica-Oblique")
    text("← row i",
         Point(SM_inset.right.x + 8, row_center_y),
         halign=:left, valign=:middle)

    # bar chart of attention weights for row i (varying heights)
    bar_center = Point(-180, inset_y)
    bar_w = 110
    bar_h = 70
    sethue("black"); fontsize(11); fontface("Helvetica")
    text("attention weights α_i",
         Point(bar_center.x, bar_center.y - bar_h/2 - 18),
         halign=:center, valign=:middle)
    text("(row i of softmax)",
         Point(bar_center.x, bar_center.y - bar_h/2 - 4),
         halign=:center, valign=:middle)

    weights = [0.05, 0.10, 0.55, 0.15, 0.10, 0.05]
    nb = length(weights)
    bw = bar_w / nb
    for (j, w) in enumerate(weights)
        bx = bar_center.x - bar_w/2 + (j - 0.5) * bw
        bh = w * 100
        sethue(pal_softmax[1])
        box(Point(bx, bar_center.y + bar_h/2 - bh/2), bw - 3, bh,
            action = :fill)
        sethue(pal_softmax[2]); setline(1)
        box(Point(bx, bar_center.y + bar_h/2 - bh/2), bw - 3, bh,
            action = :stroke)
    end
    # baseline
    sethue("#888888"); setline(1)
    line(Point(bar_center.x - bar_w/2, bar_center.y + bar_h/2),
         Point(bar_center.x + bar_w/2, bar_center.y + bar_h/2),
         action = :stroke)
    sethue("#666666"); fontsize(9); fontface("Helvetica")
    text("token 1",
         Point(bar_center.x - bar_w/2, bar_center.y + bar_h/2 + 8),
         halign=:left, valign=:top)
    text("token n",
         Point(bar_center.x + bar_w/2, bar_center.y + bar_h/2 + 8),
         halign=:right, valign=:top)
    text("Σ α_ij = 1",
         Point(bar_center.x, bar_center.y + bar_h/2 + 24),
         halign=:center, valign=:top)

    # multiplication symbol
    sethue("black"); fontface("Helvetica-Bold"); fontsize(28)
    text("×", Point(-30, inset_y), halign=:center, valign=:middle)

    # V matrix
    drawmatrix(Point(80, inset_y), n, dv, pal_value, "V", "n × d_v")

    # equals sign
    sethue("black"); fontface("Helvetica-Bold"); fontsize(28)
    text("=", Point(180, inset_y), halign=:center, valign=:middle)

    # output row (1 × d_v)
    drawmatrix(Point(280, inset_y), 1, dv, pal_output, "o_i", "1 × d_v")

    # explanation block to the right
    sethue("black"); fontface("Helvetica-Oblique"); fontsize(12)
    explanation_x = 360
    explanation_y = inset_y - 50
    text_lines = [
        "Take row i of the softmax matrix —",
        "that's the attention weights α_i",
        "for token i.",
        "",
        "Multiply by V: each row v_j of V",
        "is scaled by α_ij and the scaled rows",
        "are added together.",
        "",
        "The result is o_i = Σ_j α_ij v_j,",
        "which is row i of the output O.",
    ]
    for (k, ln) in enumerate(text_lines)
        text(ln, Point(explanation_x, explanation_y + (k-1) * 16),
             halign=:left, valign=:middle)
    end

    finish()
    return nothing
end
