type t =
  {
    mutable value : float; (* scalar value of this node calculated during forward pass *)
    mutable grad : float; (* derivative of the loss w.r.t. this node, calculated in backward pass *)
    children : t list; (* children of this node in the computation graph *)
    local_grads : float list; (* local derivative of this node w.r.t. its children *)
    mutable visited : bool; (* whether the value was already visited during backprop *)
  }

let value a = a.value

let grad a = a.grad

let make value children local_grads =
  { value; grad = 0.; children; local_grads; visited = false}

let const value =
  make value [] []

let add a b =
  make (value a +. value b) [a; b] [1.; 1.]

let sub a b =
  make (value a -. value b) [a; b] [1.; -1.]

let neg a =
  make (-. (value a)) [a] [-1.]

let cmul n a =
  make (n *. value a) [a] [n]

let mul a b =
  make (value a *. value b) [a; b] [value b; value a]

let div a b =
  make (value a /. value b) [a; b] [1. /. value b; -. value a /. (value b *. value b)]

let exp a =
  make (exp @@ value a) [a] [exp @@ value a]

(*
let pow a b =
  make (value a ** value b) [a; b] [value b *. (value a ** (value b -. 1.)); (value a ** value b) *. log (value a)]
 *)

(** Power by a constant. *)
let powc a n =
  make (value a ** n) [a] [n *. (value a ** (n -. 1.))]

let log a =
  make (log @@ value a) [a] [1. /. value a]

let relu a =
  make (max 0. (value a)) [a] [if value a > 0. then 1. else 0.]

let backward a =
  (* Breadth-first search (see the example from the Queue module). *)
  let topo a =
    let queue = Queue.create() in
    let ans = ref [] in
    Queue.push a queue;
    let rec loop () =
      if Queue.is_empty queue then !ans
      else explore @@ Queue.pop queue
    and explore a =
      if a.visited then loop () else (
        a.visited <- true;
        ans := a :: !ans;
        List.iter (fun a -> Queue.push a queue) a.children;
        loop ()
      )
    in
    loop()
  in
  let topo = List.rev @@ topo a in
  (* Printf.printf "backward: %d\n%!" (List.length topo); *)
  a.grad <- 1.;
  List.iter (fun a -> List.iter2 (fun child grad -> child.grad <- child.grad +. grad *. a.grad) a.children a.local_grads) topo

module Infix = struct
  let ( + ) = add
  let ( * ) = mul
  let ( - ) = sub
  let ( / ) = div
  let ( ** ) = powc
end

(** Vectors. *)
module Vector = struct
  (** A vector. *)
  type nonrec t = t array

  let init n f : t = Array.init n f

  (** Dimension of a vector. *)
  let dim (v:t) = Array.length v

  (** Apply a function on every coefficient. *)
  let map f (v:t) : t = Array.map f v

  (** Subvector. *)
  let subvector (v:t) off len : t = Array.sub v off len

  (** Hadamard product. *)
  let hadamard (v:t) (w:t) : t =
    assert (dim v = dim w);
    Array.init (dim v) (fun i -> mul v.(i) w.(i))

  (** Sum of the components of a vector. *)
  let sum (v:t) =
    let ans = ref @@ const 0. in
    for i = 0 to dim v - 1 do
      ans := add !ans v.(i)
    done;
    !ans

  (** Dot product of vectors. *)
  let dot v w = sum @@ hadamard v w

  (** Mutiplication by a constant. *)
  let cmul a v = map (fun x -> mul a x) v

  (** Soft max function. *)
  let soft_max (logits:t) =
    let max_val = const @@ Array.fold_left max min_float @@ Array.map value logits in
    let exps = map (fun x -> exp (sub x max_val)) logits in
    let total = sum exps in
    map (fun e -> div e total) exps

  (** RMS norm. *)
  let rms_norm (x:t) =
    let ms = sum @@ hadamard x x in
    let scale = powc (add ms (const 1e-5)) (-0.5) in
    cmul scale x

  (** Vector addition. *)
  let add (v:t) (w:t) : t =
    assert (dim v = dim w);
    Array.init (dim v) (fun i -> add v.(i) w.(i))
end

(** Matrices. *)
module Matrix = struct
  (** A matrix. *)
  type nonrec t = t array array

  (** Create a matrix with given coefficients. *)
  let init rows cols f : t =
    Array.init rows (fun i -> Array.init cols (fun j -> f i j))

  (** Number of rows. *)
  let rows a = Array.length a

  (** Number of columns. *)
  let cols a = Array.length a.(0)

  (** Get a row. *)
  let row (a:t) i : Vector.t = a.(i)

  (** List of all coefficients. *)
  let coefficients (a:t) =
    a
    |> Array.to_list
    |> List.map Array.to_list
    |> List.flatten

  (** Apply a matrix to a vector. *)
  let ap (a : t) (x : Vector.t) : Vector.t =
    Array.init (rows a) (fun i -> Vector.dot a.(i) x)

  let transpose a =
    init (cols a) (rows a) (fun i j -> a.(j).(i))
end
